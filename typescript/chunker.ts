/**
 * chunker.ts
 *
 * Section-aware chunking utility for payer policy PDFs.
 *
 * Responsibility: Take raw extracted PDF text and split it into
 * semantically meaningful chunks that preserve citation metadata
 * (source PDF, page number, section title).
 *
 * Called by: ingest.ts (one-time ingestion pipeline)
 * Output consumed by: PayerRequirementsTool.ts (via Chroma vectordb)
 *
 * Why section-aware instead of fixed-size?
 * Payer PDFs are legal documents. A fixed 500-token chunk will routinely
 * split a rule mid-clause. A bad chunk = a bad citation = a wrong billing
 * decision. We split on document structure first, enforce a size ceiling second.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface PdfPage {
  pageNumber: number;
  text: string;
}

export interface Chunk {
  content: string;
  metadata: {
    sourcePdf: string;
    pageNumber: number;
    sectionTitle: string;
    chunkIndex: number;
    totalChunks: number;
  };
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const MAX_TOKENS_PER_CHUNK = 600;

const estimateTokens = (text: string): number =>
  Math.ceil(text.trim().split(/\s+/).length * 1.3);

// ---------------------------------------------------------------------------
// Section Detection
// ---------------------------------------------------------------------------

const SECTION_HEADING_PATTERNS: RegExp[] = [
  /^\d+(\.\d+)*\s+[A-Z].{3,}/,
  /^Section\s+(I{1,3}|IV|V|VI{0,3}|IX|X)\b.*/i,
  /^[A-Z]\.\s+[A-Z].{3,}/,
  /^[A-Z][A-Z\s]{5,}$/,
  /^.{5,60}:$/,
];

const isSectionHeading = (line: string): boolean => {
  const trimmed = line.trim();
  if (!trimmed || trimmed.length < 4) return false;
  return SECTION_HEADING_PATTERNS.some((pattern) => pattern.test(trimmed));
};

// ---------------------------------------------------------------------------
// Core Chunking Logic
// ---------------------------------------------------------------------------

const splitSectionIntoChunks = (
  sectionText: string,
  sectionTitle: string,
  pageNumber: number,
  sourcePdf: string,
): Chunk[] => {
  const trimmed = sectionText.trim();
  if (!trimmed) return [];

  if (estimateTokens(trimmed) <= MAX_TOKENS_PER_CHUNK) {
    return [
      {
        content: trimmed,
        metadata: { sourcePdf, pageNumber, sectionTitle, chunkIndex: 0, totalChunks: 1 },
      },
    ];
  }

  const paragraphs = trimmed
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter((p) => p.length > 0);

  const subChunks: string[] = [];
  let currentChunk = "";

  for (const paragraph of paragraphs) {
    const combined = currentChunk ? `${currentChunk}\n\n${paragraph}` : paragraph;

    if (estimateTokens(combined) <= MAX_TOKENS_PER_CHUNK) {
      currentChunk = combined;
    } else {
      if (currentChunk) subChunks.push(currentChunk);
      currentChunk = paragraph;
    }
  }

  if (currentChunk) subChunks.push(currentChunk);

  const totalChunks = subChunks.length;

  return subChunks.map((content, index) => ({
    content,
    metadata: { sourcePdf, pageNumber, sectionTitle, chunkIndex: index, totalChunks },
  }));
};

// ---------------------------------------------------------------------------
// Page-to-Section Mapper
// ---------------------------------------------------------------------------

const buildLinePageMap = (pages: PdfPage[]): Map<number, number> => {
  const map = new Map<number, number>();
  let lineIndex = 0;

  for (const page of pages) {
    const lines = page.text.split("\n");
    for (let i = 0; i < lines.length; i++) {
      map.set(lineIndex + i, page.pageNumber);
    }
    lineIndex += lines.length;
  }

  return map;
};

// ---------------------------------------------------------------------------
// Main Export
// ---------------------------------------------------------------------------

export const chunkPdf = (pages: PdfPage[], sourcePdf: string): Chunk[] => {
  const allLines: string[] = [];
  for (const page of pages) {
    allLines.push(...page.text.split("\n"));
  }

  const linePageMap = buildLinePageMap(pages);

  const chunks: Chunk[] = [];
  let currentSectionTitle = "Preamble";
  let currentSectionLines: string[] = [];
  let currentSectionStartLine = 0;

  // Fix 1: remove unused `upToLine` parameter — flushSection uses closure vars only
  const flushSection = () => {
    const sectionText = currentSectionLines.join("\n");
    const pageNumber = linePageMap.get(currentSectionStartLine) ?? 1;
    const sectionChunks = splitSectionIntoChunks(
      sectionText,
      currentSectionTitle,
      pageNumber,
      sourcePdf,
    );
    chunks.push(...sectionChunks);
  };

  for (let i = 0; i < allLines.length; i++) {
    // Fix 2 & 3: noUncheckedIndexedAccess — guard against undefined before use
    const line = allLines[i];
    if (line === undefined) continue;

    if (isSectionHeading(line)) {
      if (currentSectionLines.length > 0) {
        flushSection();
      }
      currentSectionTitle = line.trim();
      currentSectionLines = [];
      currentSectionStartLine = i;
    } else {
      currentSectionLines.push(line);
    }
  }

  if (currentSectionLines.length > 0) {
    flushSection();
  }

  return chunks;
};

/**
 * formatCitation
 *
 * Produces the human-readable citation string the agent includes in output.
 * Example: "BCBS_Preventive_Care_Policy.pdf | Section: 4.2 Modifier Requirements | Page: 12"
 */
export const formatCitation = (chunk: Chunk): string => {
  const { sourcePdf, sectionTitle, pageNumber } = chunk.metadata;
  return `${sourcePdf} | Section: ${sectionTitle} | Page: ${pageNumber}`;
};