/**
 * PayerRequirementsTool.ts
 *
 * The "Denial Predictor" — core MCP tool for the Claims Agent.
 *
 * Given a CPT code and payer, this tool:
 *   1. Checks cms_rules.json for CMS baseline rules (fast, deterministic)
 *   2. Queries Chroma vectordb for payer-specific PDF evidence (cited)
 *   3. Returns a structured response the agent uses to:
 *      - Decide whether to apply modifiers
 *      - Flag hard stops before submission
 *      - Cite specific PDF sections in its output
 *
 * Follows the exact IMcpTool pattern from PatientAgeTool.ts and PatientIdTool.ts.
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp";
import { Request } from "express";
import { IMcpTool } from "../IMcpTool";
import { z } from "zod";
import { FhirUtilities } from "../fhir-utilities";
import { FhirClientInstance } from "../fhir-client";
import { McpUtilities } from "../mcp-utilities";
import { NullUtilities } from "../null-utilities";
import { fhirR4 } from "@smile-cdr/fhirts";
import * as fs from "fs";
import * as path from "path";
import { formatCitation, Chunk } from "../chunker";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface DenialTrigger {
  trigger: string;
  severity: "high" | "medium" | "low";
  logic_type: "hard_stop" | "documentation_gap" | "modifier_required" | "informational";
  action: string;
}

interface Modifier {
  modifier: string;
  description: string;
  condition: string;
  required: boolean;
  auto_apply: boolean;
}

interface CptRule {
  description: string;
  category: string;
  eligibility: {
    age_min: number;
    age_max: number | null;
    gender: string;
    patient_status: string;
    frequency: string;
  };
  common_modifiers: Modifier[];
  denial_triggers: DenialTrigger[];
  requires_prior_auth: boolean;
  commonly_billed_with: string[];
  notes: string;
}

interface CmsRules {
  meta: Record<string, unknown>;
  cpt_codes: Record<string, CptRule>;
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const CMS_RULES_PATH = path.resolve(__dirname, "../rules/cms_rules.json");
const VECTORDB_DIR = path.resolve(__dirname, "../vectordb");
const CHROMA_COLLECTION_NAME = "payer_policies";

/**
 * Number of PDF chunks to retrieve from Chroma per query.
 * 3 gives enough context for a citation without overwhelming the agent.
 */
const TOP_K_RESULTS = 3;

// ---------------------------------------------------------------------------
// CMS Rules Loader
// ---------------------------------------------------------------------------

/**
 * Loads cms_rules.json once at startup and caches it.
 * Synchronous read is acceptable here — this runs at server init time.
 */
let cmsRulesCache: CmsRules | null = null;

const getCmsRules = (): CmsRules => {
  if (!cmsRulesCache) {
    const raw = fs.readFileSync(CMS_RULES_PATH, "utf-8");
    cmsRulesCache = JSON.parse(raw) as CmsRules;
  }
  return cmsRulesCache;
};

// ---------------------------------------------------------------------------
// Chroma Query
// ---------------------------------------------------------------------------

/**
 * Queries Chroma for payer-specific PDF evidence relevant to a CPT code.
 *
 * CRITICAL: Must use the same embedding model as ingest.ts
 * (Xenova/all-MiniLM-L6-v2) — mismatched models produce meaningless scores.
 *
 * Returns TOP_K_RESULTS chunks with their citation metadata.
 */
let embedder: any = null;

const getEmbedding = async (text: string): Promise<number[]> => {
  if (!embedder) {
    const { pipeline } = await import("@xenova/transformers");
    embedder = await pipeline(
      "feature-extraction",
      "Xenova/all-MiniLM-L6-v2",
    );
  }

  const output = await embedder(text, {
    pooling: "mean",
    normalize: true,
  });

  return Array.from(output.data) as number[];
};

const queryChroma = async (
  cptCode: string,
  payerName: string,
): Promise<{ content: string; citation: string }[]> => {

  try {
    const { ChromaClient } = await import("chromadb");

    const client = new ChromaClient({ host: "localhost", port: 8000 });

    // Check if collection exists
    const collections = await client.listCollections();
    const collectionExists = collections.some(
      (c: any) => c.name === CHROMA_COLLECTION_NAME,
    );

    if (!collectionExists) {
      return [];
    }

    const { IncludeEnum } = await import("chromadb");
const collection = await client.getCollection({
  name: CHROMA_COLLECTION_NAME,
  embeddingFunction: undefined,
  metadata: { description: "Payer policy PDF chunks for denial prediction" },
});

    // Build a natural language query that captures what we're looking for
    // e.g. "BCBS modifier requirements for CPT 99396 preventive visit"
    const queryText = `${payerName} requirements rules coverage for CPT code ${cptCode}`;
    const queryEmbedding = await getEmbedding(queryText);

    const results = await collection.query({
      queryEmbeddings: [queryEmbedding],
      nResults: TOP_K_RESULTS,
      include: ["documents", "metadatas"] as any,
    });

    if (!results.documents?.[0]) return [];

    // Map results into { content, citation } pairs
    return results.documents[0].map((doc: string, index: number) => {
      const metadata = results.metadatas[0][index] as {
        sourcePdf: string;
        pageNumber: number;
        sectionTitle: string;
        chunkIndex: number;
        totalChunks: number;
      };

      const chunk: Chunk = {
        content: doc,
        metadata,
      };

      return {
        content: doc,
        citation: formatCitation(chunk),
      };
    });
  } catch {
    // Chroma query failure should never crash the MCP server
    return [];
  }
};

// ---------------------------------------------------------------------------
// Patient Context Helpers
// ---------------------------------------------------------------------------

/**
 * Fetches patient age from FHIR to run eligibility checks.
 * Mirrors the pattern from PatientAgeTool.ts.
 */
const getPatientAgeFromFhir = async (
  req: Request,
  patientId: string,
): Promise<number | null> => {
  try {
    const { differenceInYears, parseISO } = await import("date-fns");
    const patient = await FhirClientInstance.read<fhirR4.Patient>(
      req,
      `Patient/${patientId}`,
    );

    if (!patient?.birthDate) return null;

    const date = parseISO(patient.birthDate);
    return differenceInYears(new Date(), date);
  } catch {
    return null;
  }
};

// ---------------------------------------------------------------------------
// Eligibility Check
// ---------------------------------------------------------------------------

/**
 * Runs CMS eligibility rules against patient age.
 * Returns a denial trigger if the patient doesn't qualify, null if they do.
 */
const checkEligibility = (
  rule: CptRule,
  patientAge: number | null,
): DenialTrigger | null => {
  if (patientAge === null) return null; // Can't check without age — skip

  const { age_min, age_max } = rule.eligibility;

  if (patientAge < age_min || (age_max !== null && patientAge > age_max)) {
    return {
      trigger: `Patient age ${patientAge} is outside the eligible range (${age_min}–${age_max ?? "∞"}) for this code`,
      severity: "high",
      logic_type: "hard_stop",
      action: `Verify patient DOB and select the correct age-banded CPT code.`,
    };
  }

  return null;
};

// ---------------------------------------------------------------------------
// Response Formatter
// ---------------------------------------------------------------------------

/**
 * Formats the final structured response string.
 *
 * This is what the agent reads and quotes to the user.
 * Designed to be parseable as plain text — no JSON in the response
 * so the agent can embed it naturally in a clinical billing note.
 */
const formatResponse = (params: {
  cptCode: string;
  payerName: string;
  cmsRule: CptRule;
  eligibilityIssue: DenialTrigger | null;
  pdfEvidence: { content: string; citation: string }[];
}): string => {
  const { cptCode, payerName, cmsRule, eligibilityIssue, pdfEvidence } = params;

  const lines: string[] = [];

  // Header
  lines.push(`=== PAYER REQUIREMENTS: CPT ${cptCode} | Payer: ${payerName} ===`);
  lines.push(`Description: ${cmsRule.description}`);
  lines.push(`Category: ${cmsRule.category}`);
  lines.push(`Prior Auth Required: ${cmsRule.requires_prior_auth ? "YES" : "No"}`);
  lines.push("");

  // Eligibility
  lines.push("--- ELIGIBILITY ---");
  if (eligibilityIssue) {
    lines.push(`[HARD STOP] ${eligibilityIssue.trigger}`);
    lines.push(`Action: ${eligibilityIssue.action}`);
  } else {
    lines.push(
      `Eligible: Age range ${cmsRule.eligibility.age_min}–${cmsRule.eligibility.age_max ?? "∞"}, ` +
      `${cmsRule.eligibility.gender}, ${cmsRule.eligibility.patient_status} patient`,
    );
    lines.push(`Frequency: ${cmsRule.eligibility.frequency}`);
  }
  lines.push("");

  // Modifiers
  lines.push("--- MODIFIERS ---");
  if (cmsRule.common_modifiers.length === 0) {
    lines.push("No standard modifiers for this code.");
  } else {
    for (const mod of cmsRule.common_modifiers) {
      const flag = mod.required ? "[REQUIRED]" : "[OPTIONAL]";
      const autoFlag = mod.auto_apply ? " [AUTO-APPLY]" : "";
      lines.push(`Modifier ${mod.modifier} ${flag}${autoFlag}`);
      lines.push(`  Condition: ${mod.condition}`);
    }
  }
  lines.push("");

  // Denial Triggers — sorted by severity (high first)
  lines.push("--- DENIAL TRIGGERS ---");
  const severityOrder = { high: 0, medium: 1, low: 2 };
  const sortedTriggers = [...cmsRule.denial_triggers].sort(
    (a, b) => severityOrder[a.severity] - severityOrder[b.severity],
  );

  for (const dt of sortedTriggers) {
    const severityLabel =
      dt.severity === "high"
        ? "🔴 HIGH"
        : dt.severity === "medium"
        ? "🟡 MEDIUM"
        : "🟢 LOW";

    lines.push(`${severityLabel} | ${dt.logic_type.toUpperCase()} | ${dt.trigger}`);
    lines.push(`  Action: ${dt.action}`);
  }
  lines.push("");

  // Payer-specific PDF evidence
  lines.push("--- PAYER-SPECIFIC EVIDENCE ---");
  if (pdfEvidence.length === 0) {
    lines.push(
      `No payer-specific PDF evidence found for ${payerName}. ` +
      `Add ${payerName} policy PDFs to the pdfs/ directory and re-run ingest.ts.`,
    );
  } else {
    for (let i = 0; i < pdfEvidence.length; i++) {
      lines.push(`Evidence ${i + 1}:`);
      lines.push(`  Citation: ${pdfEvidence[i].citation}`);
      lines.push(`  Excerpt: ${pdfEvidence[i].content.substring(0, 300).trim()}...`);
      lines.push("");
    }
  }

  // Commonly billed with
  if (cmsRule.commonly_billed_with.length > 0) {
    lines.push("--- COMMONLY BILLED WITH ---");
    lines.push(cmsRule.commonly_billed_with.join(", "));
    lines.push("");
  }

  // Notes
  if (cmsRule.notes) {
    lines.push("--- NOTES ---");
    lines.push(cmsRule.notes);
  }

  return lines.join("\n");
};

// ---------------------------------------------------------------------------
// Tool Registration
// ---------------------------------------------------------------------------

class PayerRequirementsTool implements IMcpTool {
  registerTool(server: McpServer, req: Request) {
    server.registerTool(
      "GetPayerRequirements",
      {
        description:
          "Retrieves payer-specific billing requirements, modifier rules, and denial risk " +
          "for a given CPT code. Combines CMS baseline rules with payer PDF evidence. " +
          "Use this before finalizing any claim to run a pre-flight denial check.",
        inputSchema: {
          cptCode: z
            .string()
            .describe(
              "The CPT procedure code to check, e.g. '99396' or '96127'",
            )
            .nonempty(),
          payerName: z
            .string()
            .describe(
              "The patient's insurance payer, e.g. 'BCBS', 'Cigna', 'Aetna'. " +
              "Used to filter payer-specific PDF evidence from the vectordb.",
            )
            .nonempty(),
          patientId: z
            .string()
            .describe(
              "The patient's FHIR ID. Optional if patient context already exists. " +
              "Used to run age-based eligibility checks against CMS rules.",
            )
            .optional(),
        },
      },
      async ({ cptCode, payerName, patientId }) => {
        // Resolve patient ID from context if not provided
        if (!patientId) {
          patientId =
            FhirUtilities.getPatientIdIfContextExists(req) ?? undefined;
        }

        // Load CMS baseline rules
        let cmsRules: CmsRules;
        try {
          cmsRules = getCmsRules();
        } catch {
          return McpUtilities.createTextResponse(
            "ERROR: Could not load cms_rules.json. Verify the file exists at rules/cms_rules.json.",
            { isError: true },
          );
        }

        // Look up the CPT code
        const cmsRule = cmsRules.cpt_codes[cptCode];
        if (!cmsRule) {
          return McpUtilities.createTextResponse(
            `CPT code ${cptCode} is not in the CMS rules database. ` +
            `Currently supported codes: ${Object.keys(cmsRules.cpt_codes).join(", ")}. ` +
            `Add the code to rules/cms_rules.json to enable denial prediction.`,
            { isError: true },
          );
        }

        // Run eligibility check if we have a patient ID
        let eligibilityIssue: DenialTrigger | null = null;
        if (patientId) {
          const patientAge = await getPatientAgeFromFhir(req, patientId);
          eligibilityIssue = checkEligibility(cmsRule, patientAge);
        }

        // Query Chroma for payer-specific PDF evidence
        const pdfEvidence = await queryChroma(cptCode, payerName);

        // Format and return the structured response
        const response = formatResponse({
          cptCode,
          payerName,
          cmsRule,
          eligibilityIssue,
          pdfEvidence,
        });

        return McpUtilities.createTextResponse(response);
      },
    );
  }
}

export const PayerRequirementsToolInstance = new PayerRequirementsTool();