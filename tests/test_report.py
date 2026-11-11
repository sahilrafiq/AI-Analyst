"""Tests for report.py: Markdown -> Word/HTML conversion and the number checks."""
import unittest

import matplotlib.pyplot as plt

import report as R
from _common import InTempDir

MD = """# Sales report
## Summary
- Revenue was **12,345.68** in total, up *12.5%* on last year
- `North` led all regions
1. first
2. second

## Findings
Plain paragraph that
continues on a second line.

| Region | Revenue |
|---|---|
| North | 5,000 |
| South | 7,345.68 |

![Revenue by region](chart.png)
![Gone](nothere.png)
"""


class ConversionTests(InTempDir):
    def setUp(self):
        super().setUp()
        plt.figure()
        plt.bar(["a", "b"], [3, 5])
        plt.savefig("chart.png")
        plt.close()

    def test_markdown_blocks(self):
        blocks = R.parse_markdown(MD)
        self.assertEqual([b[0] for b in blocks], ["h", "h", "ul", "ol", "h", "p", "table", "img", "img"])
        self.assertEqual(blocks[5][1], "Plain paragraph that continues on a second line.")
        self.assertEqual(blocks[6][1], ["Region", "Revenue"])
        self.assertEqual(blocks[6][2][1], ["South", "7,345.68"])

    def test_word_document_structure(self):
        from docx import Document

        missing = R.to_docx(MD, "r.docx", title="Sales report", base_dir=".")
        self.assertEqual(missing, ["nothere.png"])
        doc = Document("r.docx")
        styles = [p.style.name for p in doc.paragraphs]
        self.assertEqual((styles.count("List Bullet"), styles.count("List Number")), (2, 2))
        self.assertEqual(len(doc.tables), 1)
        self.assertEqual(doc.tables[0].rows[2].cells[1].text, "7,345.68")
        self.assertEqual(len(doc.inline_shapes), 1)  # the chart is embedded
        self.assertEqual(sum(1 for p in doc.paragraphs if p.text == "Sales report"), 1)  # title not repeated
        self.assertIn("12,345.68", [r.text for p in doc.paragraphs for r in p.runs if r.bold])

    def test_html_document(self):
        missing = R.to_html(MD, "r.html", title="Sales report", base_dir=".")
        self.assertEqual(missing, ["nothere.png"])
        html = (self.tmp / "r.html").read_text(encoding="utf-8")
        for fragment in ("data:image/png;base64,", "<strong>12,345.68</strong>", "<table>", "<li>"):
            self.assertIn(fragment, html)

    def test_markdown_escapes_are_removed_everywhere(self):
        from docx import Document

        md = "## Notes\nColumn Age\\_Group and file `sales\\_summary.xlsx` and **bold\\_name**.\n"
        R.to_docx(md, "e.docx", title="T")
        R.to_html(md, "e.html", title="T")
        text = "\n".join(p.text for p in Document("e.docx").paragraphs)
        for wanted in ("Age_Group", "sales_summary.xlsx", "bold_name"):
            self.assertIn(wanted, text)
        self.assertNotIn("\\_", text)
        self.assertNotIn("\\_", (self.tmp / "e.html").read_text(encoding="utf-8"))


class NumberCheckTests(unittest.TestCase):
    SOURCE = "Total revenue: 12345.678\nmean 0.1253\np=0.0004 d=-0.007 n=400 groups 3"

    def test_numbers_that_exist_in_the_output_pass(self):
        ok = "Revenue was 12,345.68 (growth 12.5%), p-value 0.0004, d = -0.007, n=400 over 2025."
        self.assertEqual(R.unsupported_numbers(ok, self.SOURCE), [])

    def test_invented_numbers_are_flagged_small_counts_years_and_image_paths_are_not(self):
        bad = "Revenue was 99,999 and growth hit 37.2%, n=400, in 2025, with 3 groups. ![x](a_123.png)"
        self.assertEqual(R.unsupported_numbers(bad, self.SOURCE), ["99,999", "37.2%"])

    OUTPUT = """Sales: 113,036 rows x 18 columns; 1,000 duplicate rows
- Customer_Age [int64] missing 0.0%, unique 70, min 17, median 35, mean 35.92, max 87, sd 11.02
- Unit_Cost [int64] missing 0.0%, unique 130, min 1, median 9, mean 267.3, max 5,000, sd 549.8
- Unit_Price [int64] missing 0.0%, unique 150, min 2, median 120, mean 452.9, max 3,578, sd 922.1
- Revenue [int64] missing 0.0%, unique 9,000, min 2, median 270, mean 754.4, max 58,074, sd 1,309
  Australia: n=22830 mean=732.8 sd=1,075 median=398.1"""

    def test_numbers_under_the_wrong_label_are_flagged(self):
        report = ("Customer age shows a mean of 267.3 with a standard deviation of 549.8.\n"
                  "Unit price has a mean of 452.9 and a maximum value of 3,578. Revenue per transaction averages 754.4.\n"
                  "The dataset contains 113,036 rows representing sales records, and 1,000 duplicate rows.\n"
                  "Group Australia had mean 732.8.")
        flags = R.mislabeled_numbers(report, self.OUTPUT)
        self.assertEqual(len(flags), 2)
        self.assertTrue(all("267.3" in f or "549.8" in f for f in flags))
        self.assertEqual(R.mislabeled_numbers("Customer age averages 35.92 (sd 11.02).", self.OUTPUT), [])


if __name__ == "__main__":
    unittest.main()
