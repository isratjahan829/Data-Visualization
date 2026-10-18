const d = require('docx');
const { Paragraph, TextRun, Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType, AlignmentType, HeadingLevel, ImageRun } = d;
const fs = require('fs');

// Inline markup: **bold**, *italic*, [[placeholder]] (highlighted), ^sup^
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\[\[[^\]]+\]\]|\^[^^]+\^)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith('**')) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith('[[')) out.push(new TextRun({ text: '[' + t.slice(2, -2) + ']', highlight: 'yellow', bold: true, ...base }));
    else if (t.startsWith('^')) out.push(new TextRun({ text: t.slice(1, -1), superScript: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text, opts.run || {}), spacing: { after: 160, line: 360 }, alignment: opts.align, ...(opts.p || {}) });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)], spacing: { before: 320, after: 160 } });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)], spacing: { before: 240, after: 120 } });
const Bullet = (t, level = 0) => new Paragraph({ children: runs(t), numbering: { reference: 'bul', level }, spacing: { after: 80, line: 320 } });
const Num = (t, ref = 'num') => new Paragraph({ children: runs(t), numbering: { reference: ref, level: 0 }, spacing: { after: 80, line: 320 } });

const border = { style: BorderStyle.SINGLE, size: 4, color: 'A6A6A6' };
const borders = { top: border, bottom: border, left: border, right: border };
function table(header, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (txt, w, isH, shade) => new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    shading: isH ? { fill: 'E7EEF7', type: ShadingType.CLEAR, color: 'auto' } : shade ? { fill: shade, type: ShadingType.CLEAR, color: 'auto' } : undefined,
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: [new Paragraph({ children: runs(String(txt), { size: opts.size || 18, bold: isH || undefined }) })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, widths[i], true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, widths[i], false, opts.highlightRow && opts.highlightRow(r) ? 'FFF4E5' : null)) }))],
  });
}
const Caption = (t) => new Paragraph({ children: runs(t, { size: 18 }), spacing: { before: 100, after: 240 } });
function Img(path, w, h) {
  return new Paragraph({ alignment: AlignmentType.CENTER, children: [new ImageRun({ type: 'png', data: fs.readFileSync(path), transformation: { width: w, height: h } })] });
}
const numbering = { config: [
  { reference: 'bul', levels: [{ level: 0, format: d.LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
                               { level: 1, format: d.LevelFormat.BULLET, text: '–', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 270 } } } }] },
  ...['num', 'num2', 'num3', 'num4', 'refs'].map((r) => ({ reference: r, levels: [{ level: 0, format: d.LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 400 } } } }] })),
] };
const styles = {
  default: { document: { run: { font: 'Times New Roman', size: 22 } } },
  paragraphStyles: [
    { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 28, bold: true, font: 'Times New Roman' }, paragraph: { outlineLevel: 0 } },
    { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 24, bold: true, italics: true, font: 'Times New Roman' }, paragraph: { outlineLevel: 1 } },
  ],
};
module.exports = { d, P, H1, H2, Bullet, Num, table, Caption, Img, numbering, styles, runs };
