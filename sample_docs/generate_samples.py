"""Generate 2-3 short FAKE company PDFs for the out-of-the-box demo.

All content is fictional ("Muster GmbH" = German for "Sample Ltd.").
Run:  python sample_docs/generate_samples.py
Output: sample_docs/*.pdf
"""

from pathlib import Path

from fpdf import FPDF

OUT = Path(__file__).parent

FAKE_DOCS = {
    "muster-gmbh-vacation-policy.pdf": [
        (
            "Muster GmbH - Vacation Policy (FICTIONAL DEMO DOCUMENT)",
            [
                "This document is fictional and exists only to demonstrate the "
                "Business Knowledge Assistant. Muster GmbH is not a real company.",
                "1. Every full-time employee of Muster GmbH receives 25 vacation "
                "days per calendar year. Part-time employees receive a proportional "
                "number of days based on their contracted hours.",
                "2. Vacation requests must be submitted at least 14 days in advance "
                "through the HR portal. Requests for more than 10 consecutive days "
                "require approval from the direct manager and HR.",
                "3. Unused vacation days expire on March 31 of the following year. "
                "A maximum of 5 days can be carried over with written approval.",
            ],
        ),
        (
            "Vacation Policy - Continued (FICTIONAL)",
            [
                "4. During the probation period (first 6 months), employees may take "
                "up to 8 vacation days, subject to manager approval.",
                "5. Public holidays in the state of Bavaria are paid days off and do "
                "not count against the vacation balance.",
                "6. Questions about this policy: contact hr@muster-gmbh.example.",
            ],
        ),
    ],
    "muster-gmbh-it-security-policy.pdf": [
        (
            "Muster GmbH - IT Security Policy (FICTIONAL DEMO DOCUMENT)",
            [
                "This document is fictional and exists only to demonstrate the "
                "Business Knowledge Assistant. Muster GmbH is not a real company.",
                "1. Passwords must be at least 12 characters long and include "
                "uppercase letters, lowercase letters, numbers, and special "
                "characters. Passwords expire every 90 days.",
                "2. Two-factor authentication (2FA) is mandatory for all company "
                "accounts, including email, VPN, and the HR portal.",
                "3. Company laptops must be encrypted with BitLocker (Windows) or "
                "FileVault (macOS). Unencrypted USB drives are prohibited.",
            ],
        ),
        (
            "IT Security Policy - Continued (FICTIONAL)",
            [
                "4. Phishing emails must be reported to security@muster-gmbh.example "
                "within 24 hours. Do not click links in suspicious emails.",
                "5. Remote work is permitted only over the company VPN. Public Wi-Fi "
                "may be used only with the VPN active.",
                "6. Software installations require administrator approval from the "
                "IT department.",
            ],
        ),
    ],
    "muster-gmbh-widgetpro-manual.pdf": [
        (
            "WidgetPro 3000 - Product Manual (FICTIONAL DEMO DOCUMENT)",
            [
                "This document is fictional and exists only to demonstrate the "
                "Business Knowledge Assistant. The WidgetPro 3000 is not a real product.",
                "1. The WidgetPro 3000 is a portable measurement device with a "
                "battery life of up to 48 hours in standard mode and 12 hours in "
                "high-precision mode.",
                "2. Charging: use only the supplied USB-C charger. A full charge "
                "takes approximately 2 hours. The LED turns green when charging "
                "is complete.",
                "3. The device is water-resistant to IP54. Do not submerge it in "
                "water.",
            ],
        ),
        (
            "WidgetPro 3000 - Troubleshooting (FICTIONAL)",
            [
                "4. If the device does not turn on, hold the power button for 10 "
                "seconds to perform a hard reset.",
                "5. The 2-year warranty covers manufacturing defects but not "
                "physical damage or water damage beyond IP54.",
                "6. Support: support@muster-gmbh.example, Mon-Fri 9:00-17:00 CET.",
            ],
        ),
    ],
}


def make_pdf(filename: str, pages: list[tuple[str, list[str]]]) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    for title, paragraphs in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.multi_cell(0, 10, title)
        pdf.ln(4)
        pdf.set_font("Helvetica", "", 11)
        for para in paragraphs:
            pdf.multi_cell(0, 7, para)
            pdf.ln(3)
    pdf.output(str(OUT / filename))
    print(f"wrote {filename}")


if __name__ == "__main__":
    for filename, pages in FAKE_DOCS.items():
        make_pdf(filename, pages)
    print("Done. All documents are fictional demo content.")
