/**
 * ingest.ts
 *
 * One-time ingestion pipeline: Payer PDFs → Chroma vectordb.
 *
 * Run this script ONCE (or re-run when you add new PDFs) to build
 * the local Chroma database that PayerRequirementsTool.ts queries at runtime.
 *
 * This script is NEVER called by the live MCP server. It is a build-time
 * utility only. The MCP server only reads from Chroma — it never writes.
 *
 * Usage:
 *   npx ts-node ingest.ts
 *
 * Prerequisites:
 *   npm install pdfjs-dist chromadb @xenova/transformers
 *
 * Output:
 *   ./vectordb/  — Chroma persistent storage (gitignored)
 */

import * as fs from "fs";
import * as path from "path";
import { chunkPdf, PdfPage, Chunk } from "./chunker";

// Minimal type for pdfjs text items
interface TextItem {
  str: string;
  transform: number[];
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const PDFS_DIR = path.resolve(__dirname, "./pdfs");
const VECTORDB_DIR = path.resolve(__dirname, "./vectordb");
const CHROMA_COLLECTION_NAME = "payer_policies";

// ---------------------------------------------------------------------------
// PDF Text Extraction
// ---------------------------------------------------------------------------

/**
 * Extracts text from a PDF file, page by page.
 * Returns PdfPage[] which chunker.ts expects.
 *
 * Uses pdfjs-dist top-level import (compatible with v3 and v4).
 * Per-page extraction is required for accurate citation page numbers.
 */
const extractPagesFromPdf = async (filePath: string): Promise<PdfPage[]> => {
  const pdfjsLib = await import("pdfjs-dist");

  // Disable the worker — required for Node.js environments
  pdfjsLib.GlobalWorkerOptions.workerSrc = "";

  const fileBuffer = fs.readFileSync(filePath);
  const uint8Array = new Uint8Array(fileBuffer);

  const loadingTask = pdfjsLib.getDocument({
    data: uint8Array,
    useWorkerFetch: false,
    isEvalSupported: false,
    useSystemFonts: true,
  });

  const pdfDocument = await loadingTask.promise;
  const pages: PdfPage[] = [];

  for (let pageNum = 1; pageNum <= pdfDocument.numPages; pageNum++) {
    const page = await pdfDocument.getPage(pageNum);
    const textContent = await page.getTextContent();
    const pageText = reassemblePageText(textContent.items as TextItem[]);

    pages.push({
      pageNumber: pageNum,
      text: pageText,
    });
  }

  return pages;
};

/**
 * Reassembles pdfjs text items into human-readable lines.
 *
 * pdfjs returns individual positioned text fragments. Items on the same
 * line share similar Y coordinates. We group and join them, then separate
 * lines with newlines — critical for section heading detection in chunker.ts.
 */
const reassemblePageText = (items: TextItem[]): string => {
  if (!items || items.length === 0) return "";

  const lineMap = new Map<number, string[]>();

  for (const item of items) {
    if (!item.str) continue;
    const yKey = Math.round((item.transform[5] ?? 0) / 2) * 2;
    if (!lineMap.has(yKey)) {
      lineMap.set(yKey, []);
    }
    lineMap.get(yKey)!.push(item.str);
  }

  const sortedKeys = Array.from(lineMap.keys()).sort((a, b) => b - a);

  return sortedKeys
    .map((key) => lineMap.get(key)!.join(" ").trim())
    .filter((line) => line.length > 0)
    .join("\n");
};

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

/**
 * Generates a text embedding using a local model (no API key required).
 * Model downloads ~25MB on first run and caches locally.
 *
 * IMPORTANT: PayerRequirementsTool.ts must use the SAME model at query time.
 */
let embedder: any = null;

const getEmbedding = async (text: string): Promise<number[]> => {
  if (!embedder) {
    const { pipeline } = await import("@xenova/transformers");
    embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  }

  const output = await embedder(text, {
    pooling: "mean",
    normalize: true,
  });

  return Array.from(output.data) as number[];
};

// ---------------------------------------------------------------------------
// Chroma Storage
// ---------------------------------------------------------------------------

/**
 * Stores chunks in Chroma with embeddings and citation metadata.
 * Uses upsert — safe to re-run without creating duplicates.
 */
const storeChunksInChroma = async (
  chunks: Chunk[],
  pdfName: string,
): Promise<void> => {
  const { ChromaClient } = await import("chromadb");

  const client = new ChromaClient({ host: "localhost", port: 8000 });

const collection = await client.getOrCreateCollection({
  name: CHROMA_COLLECTION_NAME,
  embeddingFunction: undefined,
  metadata: { description: "Payer policy PDF chunks for denial prediction" },
});

  console.log(`  Embedding and storing ${chunks.length} chunks...`);

  const BATCH_SIZE = 10;

  for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
    const batch = chunks.slice(i, i + BATCH_SIZE);

    const ids: string[] = [];
    const embeddings: number[][] = [];
    const documents: string[] = [];
    const metadatas: Record<string, string | number>[] = [];

    for (let j = 0; j < batch.length; j++) {
      const chunk = batch[j];

      // Guard against undefined — satisfies TypeScript strict mode
      if (!chunk) continue;

      const safeSection = chunk.metadata.sectionTitle
        .replace(/[^a-zA-Z0-9]/g, "_")
        .substring(0, 50);

      const id = `${pdfName}__${safeSection}__${chunk.metadata.chunkIndex}`;
      const embedding = await getEmbedding(chunk.content);

      ids.push(id);
      embeddings.push(embedding);
      documents.push(chunk.content);
      metadatas.push({
        sourcePdf: chunk.metadata.sourcePdf,
        pageNumber: chunk.metadata.pageNumber,
        sectionTitle: chunk.metadata.sectionTitle,
        chunkIndex: chunk.metadata.chunkIndex,
        totalChunks: chunk.metadata.totalChunks,
      });
    }

    if (ids.length === 0) continue;

    await collection.upsert({ ids, embeddings, documents, metadatas });

    console.log(
      `  Stored batch ${Math.floor(i / BATCH_SIZE) + 1}/${Math.ceil(chunks.length / BATCH_SIZE)}`,
    );
  }
};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const main = async (): Promise<void> => {
  console.log("=== Payer Policy Ingestion Pipeline ===\n");

  if (!fs.existsSync(PDFS_DIR)) {
    console.error(`ERROR: pdfs/ directory not found at ${PDFS_DIR}`);
    console.error("Create the directory and add your payer PDFs before running.");
    process.exit(1);
  }

  if (!fs.existsSync(VECTORDB_DIR)) {
    fs.mkdirSync(VECTORDB_DIR, { recursive: true });
    console.log(`Created vectordb/ directory at ${VECTORDB_DIR}\n`);
  }

  const pdfFiles = fs
    .readdirSync(PDFS_DIR)
    .filter((file) => file.toLowerCase().endsWith(".pdf"));

  if (pdfFiles.length === 0) {
    console.error("ERROR: No PDF files found in pdfs/ directory.");
    process.exit(1);
  }

  console.log(`Found ${pdfFiles.length} PDF(s) to ingest:\n`);

  for (const pdfFile of pdfFiles) {
    const filePath = path.join(PDFS_DIR, pdfFile);
    console.log(`Processing: ${pdfFile}`);

    try {
      console.log("  Extracting text...");
      const pages = await extractPagesFromPdf(filePath);
      console.log(`  Extracted ${pages.length} pages`);

      console.log("  Chunking...");
      const chunks = chunkPdf(pages, pdfFile);
      console.log(`  Produced ${chunks.length} chunks`);

      await storeChunksInChroma(chunks, pdfFile);

      console.log(`  ✓ Done: ${pdfFile}\n`);
    } catch (error) {
      console.error(`  ✗ Failed to process ${pdfFile}:`, error);
    }
  }

  console.log("=== Ingestion Complete ===");
  console.log(`Chroma database written to: ${VECTORDB_DIR}`);
  console.log(`Collection name: ${CHROMA_COLLECTION_NAME}`);
  console.log("\nNext step: Start the MCP server and test PayerRequirementsTool.");
};

main().catch((error) => {
  console.error("Fatal error during ingestion:", error);
  process.exit(1);
});