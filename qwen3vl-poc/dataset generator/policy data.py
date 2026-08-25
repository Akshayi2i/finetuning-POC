#!/usr/bin/env python3
"""
generate_policy_dataset.py
==========================
Generate synthetic commercial-auto policy declaration pages modelled on a
reference "New Business" declaration, each paired with a golden JSON extraction
for fine-tuning a document-extraction model.

Each sample produces:
    * a SCANNED PDF  (rendered, then degraded: skew / noise / illumination / JPEG)
    * a GOLDEN JSON  (ground-truth extraction, identical schema across all samples)

Usage
-----
    python "policy data.py" -i "data/New Business 6.pdf" -o data --10
    python generate_policy_dataset.py -i ref.pdf -o ./out -n 250 --seed 42

Output is two files per sample, sharing one name so the pair is obvious:
    <output>/training data/sample_01.pdf    scanned PDF (image-only, no text layer)
    <output>/golden json/sample_01.json     golden JSON

The shorthand count flag works as requested: `--10` == `-n 10`. Any `--<N>` works
(`--5`, `--100`, `--2500`).

Install
-------
    pip install pillow numpy pypdfium2
    plus a renderer: weasyprint (needs GTK) OR playwright (playwright install chromium)

Note on --input
---------------
The page layout is encoded in the HTML template inside this script (see
`render_html`), which was built to match the reference declaration. `--input` is
optional: it is validated and recorded in the manifest for provenance, but the
script does NOT parse it to infer layout. Point it at your reference PDF to keep
the run traceable, or omit it entirely.
"""

import argparse
import datetime
import html as _html
import io
import json
import os
import random
import re
import sys

# ---------------------------------------------------------------------------
# dependency check
# ---------------------------------------------------------------------------
_MISSING = []
try:
    import numpy as np
    from PIL import Image, ImageFilter, ImageEnhance
except ImportError:
    _MISSING.append("pillow numpy")
try:
    import pypdfium2 as pdfium
except ImportError:
    _MISSING.append("pypdfium2")

if _MISSING:
    sys.exit("Missing dependencies: %s\n  pip install %s"
             % (", ".join(_MISSING), " ".join(_MISSING)))

# ---------------------------------------------------------------------------
# HTML -> PDF backend
#
# WeasyPrint is the reference renderer: it implements the CSS Paged Media
# margin boxes (@bottom-left / counter(page)) this template uses for the page
# footers. It needs a native GTK/Pango stack, which is trivial on Linux and
# painful on Windows.
#
# Chromium (via Playwright) needs no system libraries and renders the same
# layout, but ignores margin boxes - so the footers are re-created with its own
# header/footer template, which supplies the page numbers.
# ---------------------------------------------------------------------------
try:
    import contextlib

    # WeasyPrint prints a multi-line banner to stderr when its GTK stack is
    # missing. That is not an error here - we simply fall back to Chromium - so
    # keep it off the console.
    with contextlib.redirect_stderr(io.StringIO()):
        from weasyprint import HTML as _WeasyHTML
except Exception:                       # ImportError, or OSError from missing GTK
    _WeasyHTML = None
try:
    from playwright.sync_api import sync_playwright as _sync_playwright
except ImportError:
    _sync_playwright = None

if _WeasyHTML is None and _sync_playwright is None:
    sys.exit("No HTML->PDF renderer available. Install one:\n"
             "  pip install weasyprint          (plus the GTK3 runtime; Linux: "
             "apt install libpango-1.0-0 libpangoft2-1.0-0 libcairo2)\n"
             "  pip install playwright && playwright install chromium   (no system libs)")

RENDERER = "weasyprint" if _WeasyHTML is not None else "chromium"

# Chromium's pdf() rejects "pt"; these are the @page margins converted to inches
# (36/72, 52/72, 34/72) so both backends lay text out on the same box.
_PAGE_MARGINS = dict(top="0.5in", right="0.7222in", bottom="0.4722in", left="0.7222in")
_PAGE_FORMAT = "A4"                     # matches the reference declaration
_FOOTER_TEMPLATE = (
    '<div style="width:100%%;margin:0 52pt;font-family:Carlito,Calibri,sans-serif;'
    'font-size:6.5pt;color:#6b6b6b;display:flex;justify-content:space-between;">'
    '<span>Printed on %s at %s</span>'
    '<span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span>'
    "</div>"
)

_BROWSER = None                         # one Chromium for the whole run


def _chromium():
    """Start Chromium once and reuse it; launching per sample is ~1s wasted each."""
    global _BROWSER
    if _BROWSER is None:
        pw = _sync_playwright().start()
        _BROWSER = (pw, pw.chromium.launch())
    return _BROWSER[1]


def close_renderer():
    global _BROWSER
    if _BROWSER is not None:
        pw, browser = _BROWSER
        browser.close()
        pw.stop()
        _BROWSER = None


def html_to_pdf(html_string, printed_on, printed_at):
    """Render one HTML document to PDF bytes with whichever backend is available."""
    if RENDERER == "weasyprint":
        buf = io.BytesIO()
        _WeasyHTML(string=html_string).write_pdf(buf)
        return buf.getvalue()

    page = _chromium().new_page()
    try:
        page.set_content(html_string, wait_until="load")
        # No display_header_footer here: Chromium renders the stylesheet's own
        # @bottom-left / @bottom-right margin boxes, so adding a footer_template
        # too would draw the footer twice, overlapping.
        return page.pdf(
            format=_PAGE_FORMAT,
            prefer_css_page_size=True,      # honour @page { size: A4 } and its margins
            print_background=True,
        )
    finally:
        page.close()


# ===========================================================================
# 1. REFERENCE POOLS
# ===========================================================================
# Printed verbatim on the final page. Held here so render_html() and the golden
# JSON emit the same string.
DISCLAIMER = ("The insurance policy information contained herein is produced from data "
              "contained in the Client Management System and cannot be used to determine "
              "coverage provided by an insurance policy. Please refer to the insurance "
              "policy issued by your insurance carrier for coverage information.")

CARRIERS = [
    ("GEICO Marine Insurance Co", "35882"),
    ("Hanover American Insurance Co", "36064"),
    ("Sentry Casualty Company", "28460"),
    ("Merchants Preferred Insurance Co", "10230"),
    ("Utica Mutual Insurance Co", "25976"),
    ("Selective Way Insurance Company", "26301"),
    ("Nova Casualty Company", "42552"),
    ("Great West Casualty Company", "11371"),
    ("Berkshire Hathaway Direct Ins Co", "10391"),
    ("Cincinnati Casualty Company", "28665"),
    ("Harleysville Worcester Ins Co", "26182"),
    ("Penn Millers Insurance Company", "14982"),
]

# Coverage sets vary by state so the model learns which fields co-occur.
STATES = {
    "New York":      dict(abbr="NY", pip=True,  pip_label="Personal Injury Protection (I)",
                          extras=["suspension_um", "childcare", "apip", "lef"], factor=1.00),
    "New Jersey":    dict(abbr="NJ", pip=True,  pip_label="Personal Injury Protection",
                          extras=["umbi_pd"], factor=0.94),
    "Connecticut":   dict(abbr="CT", pip=False, pip_label=None,
                          extras=["medpay", "underinsured"], factor=0.88),
    "Pennsylvania":  dict(abbr="PA", pip=True,  pip_label="First Party Benefits",
                          extras=["medpay", "underinsured"], factor=0.83),
    "Texas":         dict(abbr="TX", pip=True,  pip_label="Personal Injury Protection",
                          extras=["umbi_pd"], factor=0.79),
    "Florida":       dict(abbr="FL", pip=True,  pip_label="Personal Injury Protection",
                          extras=["medpay"], factor=1.06),
    "Illinois":      dict(abbr="IL", pip=False, pip_label=None,
                          extras=["medpay", "underinsured"], factor=0.81),
    "Georgia":       dict(abbr="GA", pip=False, pip_label=None,
                          extras=["medpay"], factor=0.77),
    "Massachusetts": dict(abbr="MA", pip=True,  pip_label="Personal Injury Protection",
                          extras=["medpay", "underinsured"], factor=0.92),
    "Ohio":          dict(abbr="OH", pip=False, pip_label=None,
                          extras=["medpay"], factor=0.72),
}

TRUCKS = [
    ("FREIGHTLINER", "M2", 26000), ("FREIGHTLINER", "CASCADIA", 33000),
    ("ISUZU", "NPR HD", 14500), ("ISUZU", "FTR", 25950),
    ("HINO", "268", 25950), ("HINO", "L6", 33000),
    ("FORD", "F-650", 26000), ("FORD", "TRANSIT 350", 9500),
    ("INTERNATIONAL", "MV607", 25999), ("KENWORTH", "T270", 26000),
    ("PETERBILT", "337", 25999), ("RAM", "5500", 19500),
    ("CHEVROLET", "SILVERADO 3500", 14000), ("GMC", "SAVANA 3500", 9900),
    ("MERCEDES-BENZ", "SPRINTER 3500", 11030),
]

BODY_TYPES = ["Truck", "Truck", "Truck", "Van", "Box Truck", "Tractor"]

BIZ_DESC = [
    "Other Retail Trade Operations", "Wholesale Distribution Operations",
    "Building Materials Dealer", "Food Service Distribution",
    "Contractor - Plumbing Services", "Furniture Delivery Operations",
    "Landscaping Services Operations", "HVAC Contracting Operations",
    "Appliance Sales And Service", "General Freight Trucking - Local",
]

CORP_SUFFIX = ["Services Corp", "Holdings Corp", "Enterprises Inc", "Logistics LLC",
               "Distributors Inc", "Industries Corp", "Transport Group Inc", "Supply Co"]

FIRST = ["Mary", "Erick", "Carlos", "Denise", "Rajiv", "Angela", "Marcus", "Priya",
         "Steven", "Lucia", "Nathan", "Grace", "Omar", "Tanya", "Victor", "Elena",
         "Dwayne", "Rosa", "Hector", "Janet", "Kevin", "Sofia", "Andre", "Beatriz"]
LAST = ["Mills", "Otero", "Tejeda Rosales", "Whitfield", "Ramachandran", "Boone",
        "Delgado", "Krishnan", "Vargas", "Hollingsworth", "Nakamura", "Pierre-Louis",
        "Castellano", "Ferreira", "Osei", "Lindqvist", "Barrera", "Aguilar"]
MID = list("ABCDEFGHJKLMNPRSTVW")

CITIES = ["West Jessicastad", "North Connorshire", "Port Adrianfort", "New Marcusview",
          "Lake Danielle", "East Kimberlyton", "South Terrance", "Farmingdale",
          "Hauppauge", "Bay Shore", "Elmsford", "Ronkonkoma", "Yaphank", "Deer Park"]
STREET_NAMES = ["Price Pkwy", "Katherine Parks", "Commerce Blvd", "Sunrise Hwy",
                "Executive Dr", "Marcus Ave", "Industrial Loop", "Wireless Blvd",
                "Adams Ct", "Oser Ave", "Corporate Center Dr", "Engineers Rd"]

INTEREST_TYPES = ["Additional Insured", "Loss Payee", "Lienholder", "Additional Interest"]
LENDERS = ["P.c.richard&Son Long Island Corporation", "Daimler Truck Financial Services USA",
           "Ryder Truck Rental Inc", "PACCAR Financial Corp", "Wells Fargo Equipment Finance",
           "Mitsubishi HC Capital America", "Ascentium Capital LLC", "Signature Financial LLC",
           "Isuzu Finance of America Inc"]

PAY_PLANS = [("Monthly", 11), ("Monthly", 10), ("Quarterly", 4), ("Semi-Annual", 2),
             ("Full Pay", 1), ("Ten Pay", 10), ("Monthly", 9)]
BILL_METHODS = ["Company Policy Billed", "Agency Billed", "Direct Bill - EFT",
                "Company Policy Billed", "Direct Bill - Invoice"]
PAYORS = ["Insured", "Insured", "Insured", "First Named Insured", "Additional Insured"]

VIN_CHARS = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"   # no I, O, Q — real VIN rule


# ===========================================================================
# 2. DATA GENERATION  ->  golden JSON
# ===========================================================================
def _money(x):
    return "${:,.2f}".format(x)


def _mdY(d):
    return d.strftime("%m/%d/%Y").lstrip("0").replace("/0", "/")


def build_sample(idx, rng, carrier_name=None, state_name=None, n_veh=None, n_drv=None,
                 n_int=None, n_contact=None):
    """Build one golden-JSON record. `rng` is a seeded random.Random.

    carrier_name / state_name / n_veh / n_drv pin the document's identity and shape
    so that every sample is the same declaration with different values. Leave them
    None to cycle through CARRIERS and STATES and randomise the counts instead.
    """

    def vin():
        return "".join(rng.choice(VIN_CHARS) for _ in range(17))

    def person():
        return rng.choice(FIRST), rng.choice(MID), rng.choice(LAST)

    def addr(ab):
        return dict(street="%d %s" % (rng.randint(20, 9800), rng.choice(STREET_NAMES)),
                    city=rng.choice(CITIES), state=ab,
                    zip="%05d" % rng.randint(6001, 99999))

    if carrier_name:
        match = [c for c in CARRIERS if c[0] == carrier_name]
        if not match:
            raise SystemExit("unknown carrier %r; known: %s"
                             % (carrier_name, ", ".join(c[0] for c in CARRIERS)))
        carrier, naic = match[0]
    else:
        carrier, naic = CARRIERS[idx % len(CARRIERS)]
    if state_name:
        if state_name not in STATES:
            raise SystemExit("unknown state %r; known: %s"
                             % (state_name, ", ".join(STATES)))
    else:
        state_name = list(STATES.keys())[idx % len(STATES)]
    st = STATES[state_name]
    ab, f = st["abbr"], st["factor"]

    n_veh = n_veh or rng.choice([1, 2, 2, 3, 3, 4, 5])
    n_drv = n_drv or max(1, n_veh + rng.choice([-1, 0, 0, 1, 2]))
    n_int = n_int or rng.choice([1, 1, 1, 2, 2])   # >=1 keeps the key set uniform
    n_contact = n_contact or rng.choice([1, 1, 2])

    # ---- dates -----------------------------------------------------------
    eff = datetime.date(2026, rng.randint(1, 11), rng.randint(1, 28))
    txn = eff - datetime.timedelta(days=rng.randint(3, 45))
    exp = datetime.date(eff.year + 1, eff.month, eff.day)
    printed = txn + datetime.timedelta(days=rng.randint(0, 6))
    printed_time = "%02d:%02d" % (rng.randint(8, 18), rng.choice([0, 15, 30, 45]))

    # ---- insured ---------------------------------------------------------
    _, _, corp_last = person()
    insured_name = "%s %s" % (corp_last.split()[0], rng.choice(CORP_SUFFIX))
    insured_addr = addr(ab)
    ani = []
    for _ in range(rng.choice([0, 1, 1, 2])):
        af, am, al = person()
        ani.append(rng.choice([af, "%s %s" % (af, al), "%s %s %s" % (af, am, al)]))

    policy_number = str(rng.randint(1000000000, 9999999999))
    billing_account = policy_number if rng.random() < 0.7 else str(rng.randint(10 ** 9, 10 ** 10 - 1))
    contract_number = "%s%d" % (rng.choice("BCDG"), rng.randint(10000, 99999))

    # ---- contacts --------------------------------------------------------
    ctypes = ["Business Phone", "Mobile Phone", "Email", "Fax"]
    rng.shuffle(ctypes)
    contacts = []
    for ct in ctypes[:n_contact]:
        stem = insured_name.split()[0].lower()
        info = ("%s@%scorp.com" % (stem, stem) if ct == "Email"
                else "%d-%d-%04d" % (rng.randint(201, 959), rng.randint(200, 989),
                                     rng.randint(0, 9999)))
        cname = None
        if rng.random() < 0.35:
            cf, _, cl = person()
            cname = "%s %s" % (cf, cl)
        contacts.append(dict(type=ct, information=info, contact_name=cname))

    # ---- location / garaging --------------------------------------------
    loc_addr = addr(ab)
    garaging = "%s, %s, %s %05d" % (loc_addr["street"], rng.choice(CITIES), ab,
                                    rng.randint(6001, 19999))

    # ---- vehicles --------------------------------------------------------
    vehicles = []
    for i in range(n_veh):
        mk, md, gvw = rng.choice(TRUCKS)
        vehicles.append(dict(sequence=i + 1, year=str(rng.choice([2023, 2024, 2025, 2026])),
                             make=mk, model=md, vin=vin(),
                             body_type=rng.choice(BODY_TYPES),
                             territory=str(rng.randint(101, 899)),
                             gvw="{:,.2f}".format(gvw),
                             registration_state=state_name,
                             garaging_location=garaging, _gvw=gvw))

    # ---- drivers ---------------------------------------------------------
    drivers = []
    for i in range(n_drv):
        df, dm, dl = person()
        name = "%s %s %s" % (df, dm, dl) if rng.random() < 0.5 else "%s %s" % (df, dl)
        dob = datetime.date(rng.randint(1952, 1998), rng.randint(1, 12), rng.randint(1, 28))
        drivers.append(dict(sequence=i + 1, name=name, license_state=ab,
                            license_number=str(rng.randint(10 ** 8, 10 ** 9 - 1)),
                            date_of_birth=_mdY(dob)))

    # ---- limits ----------------------------------------------------------
    csl = rng.choice([500000, 1000000, 1000000, 2000000])
    pip_limit = rng.choice([50000, 25000, 10000, 15000])
    um_limit = rng.choice(["25,000/50,000", "50,000/100,000", "100,000/300,000"])
    comp_ded = rng.choice([500, 1000, 1000, 2500])
    coll_ded = rng.choice([500, 1000, 1000, 2500])
    rental = rng.choice(["50/1,500", "75/2,250", "40/1,200"])
    medpay_limit = rng.choice([5000, 10000, 2000])
    uim_limit = rng.choice(["25,000/50,000", "100,000/300,000"])
    susp_limit = rng.choice([500000, 250000, 1000000])

    # ---- per-vehicle premium matrix -------------------------------------
    rows = []

    def add_row(desc, limit, ded, fn):
        rows.append(dict(description=desc, limit=limit, deductible=ded,
                         premiums=[fn(v) for v in vehicles]))

    cslf = csl / 1000000.0
    add_row("Combined Single Limit Liability", "{:,}".format(csl), None,
            lambda v: round((6800 + v["_gvw"] * 0.16) * f * (0.72 + 0.28 * cslf)
                            * rng.uniform(0.93, 1.09), 0))
    if st["pip"]:
        add_row(st["pip_label"], "{:,}".format(pip_limit), "0 Ded",
                lambda v: round(rng.uniform(255, 385) * f, 0))
    add_row("Comprehensive", None, "{:,} Ded".format(comp_ded),
            lambda v: round((690 + v["_gvw"] * 0.004) * f * rng.uniform(0.92, 1.1), 0))
    add_row("Collision", None, "{:,} Ded".format(coll_ded),
            lambda v: round((2450 + v["_gvw"] * 0.028) * f * rng.uniform(0.92, 1.1), 0))
    add_row("Rental Reimbursement", rental, None, lambda v: float(rng.choice([66, 72, 84])))
    add_row("Uninsured Motorist Bi Split Limit", um_limit, None,
            lambda v: float(rng.choice([78, 96, 112, 134])))

    ex = st["extras"]
    if "medpay" in ex:
        add_row("Medical Payments", "{:,}".format(medpay_limit), None,
                lambda v: float(rng.choice([44, 58, 63, 71])))
    if "underinsured" in ex:
        add_row("Underinsured Motorist Bi Split Limit", uim_limit, None,
                lambda v: float(rng.choice([52, 61, 88])))
    if "umbi_pd" in ex:
        add_row("Uninsured Motorist Property Damage", "25,000", "250 Ded",
                lambda v: float(rng.choice([21, 28, 34])))
    if "childcare" in ex:
        add_row("Childcare Expenses", "25", None, lambda v: "Included")
    if "apip" in ex:
        add_row("Apip Death Benefits", "2,000", None, lambda v: "Included")
    if "suspension_um" in ex:
        add_row("Suspension Of Uninsured Motorist Liability Coverage",
                "{:,}".format(susp_limit), None, lambda v: float(rng.choice([271, 288, 305])))
    if "lef" in ex:
        add_row("Law Enforcement Fee", None, None, lambda v: 10.0)

    col_totals = [sum(r["premiums"][i] for r in rows
                      if isinstance(r["premiums"][i], (int, float)))
                  for i in range(n_veh)]

    # ---- policy-level coverage summary ----------------------------------
    lob_rows = [dict(coverage="Combined Single Limit Liability",
                     limit="{:,}".format(csl), premium="Included")]
    if st["pip"]:
        lob_rows.append(dict(coverage=st["pip_label"].replace(" (I)", ""),
                             limit="{:,}".format(pip_limit), premium="Included"))
    lob_rows.append(dict(coverage="Uninsured Motorist Bi Split Limit",
                         limit=um_limit, premium="Included"))
    if "suspension_um" in ex:
        lob_rows.append(dict(coverage="Suspension Of Uninsured Motorist Liability Coverage",
                             limit="{:,}".format(susp_limit), premium="Included"))
    if "medpay" in ex:
        lob_rows.append(dict(coverage="Medical Payments",
                             limit="{:,}".format(medpay_limit), premium="Included"))

    # ---- additional interests drive the fee lines ------------------------
    interests = []
    for r in range(n_int):
        interests.append(dict(nature_of_interest=rng.choice(INTEREST_TYPES),
                              name=rng.choice(LENDERS), address=addr(ab),
                              rank=str(r + 1), payor=None, account_number=None))
    fee_total = 20.0 * n_int
    lob_rows.append(dict(coverage="Additional Interests", limit=None,
                         premium=_money(fee_total)))
    n_ai = sum(1 for i in interests if i["nature_of_interest"] == "Additional Insured")
    if n_ai:
        lob_rows.append(dict(coverage="Additional Insured", limit=None,
                             premium=_money(20.0 * n_ai)))
        fee_total += 20.0 * n_ai

    term_amount = sum(col_totals) + fee_total
    plan, npay = rng.choice(PAY_PLANS)

    return {
        "document_type": "Policy Declaration",
        "carrier_name": carrier,
        "transaction": {
            "transaction_type": "NEW BUSINESS",
            "transaction_date": _mdY(txn),
            "transaction_effective_date": _mdY(eff),
            "policy_type": "Business Automobile",
            "naic": naic,
            "policy_number": policy_number,
            "billing_account_number": billing_account,
            "contract_number": contract_number,
            "effective_date": _mdY(eff),
            "expiration_date": _mdY(exp),
            "term_amount": _money(term_amount),
        },
        "named_insured": {
            "entity_type": "CORPORATION",
            "name": insured_name,
            "address": insured_addr,
            "additional_named_insureds": ani,
        },
        "contact_info": contacts,
        "line_of_business": {
            "name": "Business Automobile",
            "premium": _money(term_amount),
            "inception_date": _mdY(eff),
            "effective_date": _mdY(eff),
            "net_change_amount": _money(term_amount),
            "coverages": lob_rows,
            "total": _money(fee_total),
        },
        "additional_interests": interests,
        "states": [{
            "state": state_name,
            "locations": [{
                "location_number": 1,
                "location_address": loc_addr["street"].upper(),
                "vehicle_coverages": {
                    "vehicle_columns": ["%s %s %s" % (v["year"], v["make"], v["model"])
                                        for v in vehicles],
                    "rows": [dict(description=r["description"], limit=r["limit"],
                                  deductible=r["deductible"],
                                  premiums=[p if isinstance(p, str) else _money(p)
                                            for p in r["premiums"]])
                             for r in rows],
                    "totals": [_money(t) for t in col_totals],
                },
            }],
        }],
        "vehicles": [{k: v for k, v in veh.items() if not k.startswith("_")}
                     for veh in vehicles],
        "drivers": drivers,
        "commercial_policy_information": {
            "controlling_state": state_name,
            "operation_business_description": rng.choice(BIZ_DESC),
        },
        "locations": [{"location_number": "1", "address": loc_addr,
                       "legal_description": None}],
        "pay_plan": {"payment_plan": plan, "number_of_payments": str(npay),
                     "bill_method": rng.choice(BILL_METHODS), "payor": rng.choice(PAYORS)},
        # Printed verbatim on the last page, so it belongs in the ground truth too.
        "disclaimer": DISCLAIMER,
        "document_metadata": {"printed_on": _mdY(printed), "printed_at": printed_time,
                              "total_pages": 3},
    }


# ===========================================================================
# 3. HTML / PDF RENDERING  (layout mirrors the reference declaration)
# ===========================================================================
BLUE, GRAY = "#1F6FB9", "#333333"

# Measured by rendering the SAME word ("Business Automobile", the POLICY TYPE value)
# in both documents at identical zoom: the reference's glyphs are 69px tall where this
# template produced 77px, so its type is ~12% smaller than the unscaled template.
# Scaling every font-size (and nothing else - page margins stay put) matches it.
# An earlier line-pitch estimate suggested the opposite; it was comparing lines with
# different content and wrapping, and was wrong.
FONT_SCALE = 0.96


_FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
_FONT_FACE_CACHE = None


def _font_face_css():
    """Carlito, embedded as base64 @font-face rules.

    The reference declaration was typeset in Carlito. Windows has no Carlito and
    substitutes Calibri, whose bold is noticeably heavier - which is what made the
    generated pages look emphasised next to the reference. Embedding the real font
    removes the substitution, so output is identical on any machine.
    """
    global _FONT_FACE_CACHE
    if _FONT_FACE_CACHE is not None:
        return _FONT_FACE_CACHE
    import base64

    rules = []
    for weight, name in ((400, "Carlito-Regular"), (700, "Carlito-Bold")):
        path = os.path.join(_FONT_DIR, name + ".ttf")
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as fh:
            data = base64.b64encode(fh.read()).decode("ascii")
        rules.append("@font-face{font-family:Carlito;font-style:normal;font-weight:%d;"
                     "src:url(data:font/ttf;base64,%s) format('truetype');}" % (weight, data))
    if not rules:
        print("WARNING: %s holds no Carlito TTFs; falling back to system fonts, which "
              "will not match the reference exactly" % _FONT_DIR, file=sys.stderr)
    _FONT_FACE_CACHE = "".join(rules)
    return _FONT_FACE_CACHE


def _scale_fonts(css):
    return re.sub(r"font-size:\s*(\d+(?:\.\d+)?)pt",
                  lambda m: "font-size: %.2fpt" % (float(m.group(1)) * FONT_SCALE), css)

CSS = """
@page {
  size: A4; margin: 36pt 52pt 34pt 52pt;
  @bottom-left  { content: "Printed on __PRINTED__ at __TIME__";
                  font-family: Carlito, Calibri; font-size: 6.5pt; color: #6b6b6b; }
  @bottom-right { content: "Page " counter(page) " of " counter(pages);
                  font-family: Carlito, Calibri; font-size: 6.5pt; color: #6b6b6b; }
}
body { font-family: Carlito, Calibri, "DejaVu Sans", sans-serif; font-size: 8.2pt; color: #1a1a1a; }
h1 { color: __BLUE__; font-size: 15pt; margin: 0 0 9pt 0; font-weight: bold; }
h2 { color: __BLUE__; font-size: 12.5pt; margin: 8pt 0 2pt 0; font-weight: bold;
     border-bottom: 0.6pt solid #d6d6d6; padding-bottom: 3pt; }
h2.plain { border-bottom: none; }
h3 { color: __BLUE__; font-size: 10.5pt; margin: 8pt 0 3pt 0; font-weight: bold;
     border-bottom: 0.6pt solid #d6d6d6; padding-bottom: 3pt; }
h4 { font-size: 9pt; margin: 6pt 0 3pt 0; font-weight: bold; color: #111; }
h4.ul { border-bottom: 0.6pt solid #d6d6d6; padding-bottom: 3pt; }
.sup { font-size: 5.6pt; vertical-align: super; color: #555; font-weight: normal; }
.lbl { color: __GRAY__; letter-spacing: .2px; }
.val { font-weight: bold; }
.small { font-size: 6.6pt; color: #333; }
table { width: 100%; border-collapse: collapse; }
.hdr th { color: __GRAY__; font-weight: normal; font-size: 7.6pt; text-align: left;
          border-bottom: 0.6pt solid #d6d6d6; padding: 2.5pt 4pt; }
td { padding: 2.7pt 4pt; border-bottom: 0.5pt solid #ececec; vertical-align: top; }
.num { text-align: right; }
.tot td { border-bottom: none; }
.amt { float: right; color: #111; font-size: 12.5pt; font-weight: bold; }
.vt th { font-size: 7.4pt; color: #444; font-weight: bold; text-align: right; padding: 2pt 4pt; }
.vt th.d { text-align: left; color: __GRAY__; font-weight: normal;
           border-bottom: 0.6pt solid #d6d6d6; }
.vt th.yr { border-bottom: none; }
.vt td { font-size: 7.8pt; padding: 2.1pt 4pt; }
.pblock { margin: 0 0 0.8pt 0; }
.disc { font-size: 8pt; line-height: 1.45; }
"""


def _esc(s):
    return _html.escape(str(s)) if s is not None else ""


def _kv(label, value):
    return '<div class="pblock"><span class="lbl">%s</span> <span class="val">%s</span></div>' \
           % (_esc(label), _esc(value))


def render_html(d):
    t, ni, lob = d["transaction"], d["named_insured"], d["line_of_business"]
    stblk = d["states"][0]
    loc = stblk["locations"][0]
    vc = loc["vehicle_coverages"]
    meta = d["document_metadata"]
    o = []

    # ---------------- header ----------------
    o.append('<h1>%s</h1>' % _esc(d["carrier_name"]))
    o.append('<table style="border:none"><tr><td style="border:none;width:52%;padding-left:0">')
    o.append('<div class="lbl">NAMED INSURED(S) (%s)</div>' % _esc(ni["entity_type"]))
    nm = ni["name"].split(" ", 1)
    o.append('<div><span class="val">%s</span> &nbsp;&nbsp; <span class="val">%s</span></div>'
             % (_esc(nm[0]), _esc(nm[1] if len(nm) > 1 else "")))
    a = ni["address"]
    o.append('<div class="small">%s</div>' % _esc(a["street"]))
    o.append('<div>%s, %s %s</div>' % (_esc(a["city"]), _esc(a["state"]), _esc(a["zip"])))
    if ni["additional_named_insureds"]:
        o.append('<div class="lbl" style="margin-top:10pt">ADDITIONAL NAMED INSUREDS</div>')
        for x in ni["additional_named_insureds"]:
            o.append('<div>%s</div>' % _esc(x))
    o.append('</td><td style="border:none;width:48%;padding-right:0">')
    o.append('<div style="color:%s;font-size:10.5pt;font-weight:bold;margin-bottom:4pt">%s</div>'
             % (BLUE, _esc(t["transaction_type"])))
    o.append(_kv("TRANSACTION DATE:", t["transaction_date"]))
    o.append(_kv("TRANSACTION EFFECTIVE DATE:", t["transaction_effective_date"]))
    o.append(_kv("POLICY TYPE:", t["policy_type"]))
    o.append(_kv("NAIC:", t["naic"]))
    for lab, key in [("POLICY NUMBER:", "policy_number"),
                     ("BILLING ACCOUNT #:", "billing_account_number"),
                     ("CONTRACT #:", "contract_number")]:
        o.append('<div class="pblock"><span class="lbl">%s</span> '
                 '<span class="small">%s</span></div>' % (lab, _esc(t[key])))
    o.append('<div style="height:6pt"></div>')
    o.append(_kv("EFFECTIVE DATE:", t["effective_date"]))
    o.append(_kv("EXPIRATION DATE:", t["expiration_date"]))
    o.append(_kv("TERM AMOUNT:", t["term_amount"]))
    o.append('</td></tr></table>')

    # ---------------- contact info ----------------
    o.append('<h2>Contact Info <span class="sup">%d</span></h2>' % len(d["contact_info"]))
    o.append('<table><tr class="hdr"><th style="width:34%">Type</th>'
             '<th style="width:33%">Information</th><th>Contact Name</th></tr>')
    for c in d["contact_info"]:
        o.append('<tr><td>%s</td><td>%s</td><td>%s</td></tr>'
                 % (_esc(c["type"]), _esc(c["information"]), _esc(c["contact_name"])))
    o.append('</table>')

    # ---------------- line of business ----------------
    o.append('<h2 class="plain">%s <span class="amt">%s</span></h2>'
             % (_esc(lob["name"]), _esc(lob["premium"])))
    o.append('<div style="clear:both"></div>')
    o.append(_kv("INCEPTION DATE:", lob["inception_date"]))
    o.append(_kv("EFFECTIVE DATE:", lob["effective_date"]))
    o.append(_kv("NET CHANGE AMOUNT:", lob["net_change_amount"]))
    o.append('<table style="margin-top:7pt"><tr class="hdr"><th>Coverage</th>'
             '<th class="num" style="width:22%">Limit</th>'
             '<th class="num" style="width:14%">Premium</th></tr>')
    for c in lob["coverages"]:
        o.append('<tr><td>%s</td><td class="num val">%s</td><td class="num">%s</td></tr>'
                 % (_esc(c["coverage"]), _esc(c["limit"]), _esc(c["premium"])))
    o.append('<tr class="tot"><td>Total</td><td></td><td class="num">%s</td></tr></table>'
             % _esc(lob["total"]))

    # ---------------- additional interests ----------------
    if d["additional_interests"]:
        o.append('<h4 class="ul">ADDITIONAL INTERESTS <span class="sup">%d</span></h4>'
                 % len(d["additional_interests"]))
        o.append('<table><tr class="hdr"><th style="width:20%">Nature of Interest</th>'
                 '<th style="width:34%">Name/Address</th><th style="width:9%">Rank</th>'
                 '<th style="width:12%">Payor</th><th>Account Number</th></tr>')
        for i in d["additional_interests"]:
            ia = i["address"]
            o.append('<tr><td class="val">%s</td><td><span class="val">%s</span><br>'
                     '<span class="val">%s</span><br><span class="small">%s, %s %s</span></td>'
                     '<td>%s</td><td>%s</td><td>%s</td></tr>'
                     % (_esc(i["nature_of_interest"]), _esc(i["name"]), _esc(ia["street"]),
                        _esc(ia["city"]), _esc(ia["state"]), _esc(ia["zip"]),
                        _esc(i["rank"]), _esc(i["payor"]), _esc(i["account_number"])))
        o.append('</table>')

    # ---------------- vehicle coverage matrix ----------------
    o.append('<h3>%s</h3>' % _esc(stblk["state"]))
    o.append('<h4 class="ul">LOCATION %s: %s</h4>'
             % (_esc(loc["location_number"]), _esc(loc["location_address"])))
    o.append('<h4>VEHICLE COVERAGES</h4>')
    cols = vc["vehicle_columns"]
    wid = max(9, int(46 / max(1, len(cols))))
    o.append('<table class="vt" style="width:96%"><thead>')
    o.append('<tr><th class="yr"></th><th class="yr"></th><th class="yr"></th>')
    for c in cols:
        o.append('<th class="yr" style="width:%d%%">%s</th>' % (wid, _esc(c.split(" ")[0])))
    o.append('<th class="yr" style="width:6%"></th></tr>')
    o.append('<tr><th class="yr"></th><th class="yr"></th><th class="yr"></th>')
    for c in cols:
        o.append('<th class="yr">%s</th>' % _esc(" ".join(c.split(" ")[1:])))
    o.append('<th class="yr"></th></tr>')
    o.append('<tr><th class="d" style="width:26%">Description</th>'
             '<th class="d num" style="width:13%">Limit</th>'
             '<th class="d num" style="width:11%">Deductible</th>')
    o.append('<th class="d num"></th>' * len(cols))
    o.append('<th class="d"></th></tr></thead><tbody>')
    for r in vc["rows"]:
        o.append('<tr><td class="val">%s</td><td class="num val">%s</td><td class="num val">%s</td>'
                 % (_esc(r["description"]), _esc(r["limit"]), _esc(r["deductible"])))
        for p in r["premiums"]:
            o.append('<td class="num val">%s</td>' % _esc(p))
        o.append('<td></td></tr>')
    o.append('<tr class="tot"><td>Total</td><td></td><td></td>')
    for tt in vc["totals"]:
        o.append('<td class="num">%s</td>' % _esc(tt))
    o.append('<td></td></tr></tbody></table>')

    # ---------------- page 2: vehicles / drivers ----------------
    o.append('<div style="break-before:page"></div>')
    o.append('<h4 class="ul">VEHICLES <span class="sup">%d</span></h4>' % len(d["vehicles"]))
    for v in d["vehicles"]:
        o.append('<div style="break-inside:avoid">')
        o.append('<div style="color:%s;font-weight:bold;font-size:9pt;margin-top:11pt;'
                 'border-bottom:0.6pt solid #d6d6d6;padding-bottom:3pt">'
                 '<span style="color:#111">%d.</span> %s %s %s</div>'
                 % (BLUE, v["sequence"], _esc(v["year"]), _esc(v["make"]), _esc(v["model"])))
        o.append('<table style="margin-top:5pt;border:none"><tr>')
        o.append('<td style="border:none;width:52%%;padding-left:0">%s%s%s%s%s</td>'
                 % (_kv("VIN:", v["vin"]), _kv("Body Type:", v["body_type"]),
                    _kv("Territory:", v["territory"]), _kv("GVW:", v["gvw"]),
                    _kv("Registration State:", v["registration_state"])))
        gl = v["garaging_location"].split(", ")
        o.append('<td style="border:none;padding-left:12pt">'
                 '<span class="lbl">Garaging Location:</span> '
                 '<span class="val">%s,</span><br><span class="val">%s</span></td>'
                 % (_esc(", ".join(gl[:-1])), _esc(gl[-1])))
        o.append('</tr></table></div>')

    o.append('<h4 class="ul" style="margin-top:16pt">DRIVERS <span class="sup">%d</span></h4>'
             % len(d["drivers"]))
    for dr in d["drivers"]:
        o.append('<div style="break-inside:avoid">')
        o.append('<div style="color:%s;font-weight:bold;font-size:9pt;margin-top:10pt;'
                 'border-bottom:0.6pt solid #d6d6d6;padding-bottom:3pt">'
                 '<span style="color:#111">%d.</span> %s</div>' % (BLUE, dr["sequence"],
                                                                   _esc(dr["name"])))
        o.append('<div style="margin-top:4pt">%s</div>'
                 % _kv("License State / Number:", "%s / %s" % (dr["license_state"],
                                                              dr["license_number"])))
        o.append('<div><span class="lbl">Date of Birth:</span> '
                 '<span class="small">%s</span></div></div>' % _esc(dr["date_of_birth"]))

    cpi = d["commercial_policy_information"]
    o.append('<h3>Commercial Policy Information</h3>')
    o.append('<div style="margin-top:5pt">%s%s</div>'
             % (_kv("Controlling State:", cpi["controlling_state"]),
                _kv("Operation Business Description:", cpi["operation_business_description"])))

    o.append('<h2>Locations <span class="sup">%d</span></h2>' % len(d["locations"]))
    o.append('<table><tr class="hdr"><th style="width:14%">Location #</th>'
             '<th style="width:44%">Address</th><th>Legal Description</th></tr>')
    for L in d["locations"]:
        la = L["address"]
        o.append('<tr><td>%s</td><td><span class="val">%s</span><br>'
                 '<span class="small">%s, %s %s</span></td><td>%s</td></tr>'
                 % (_esc(L["location_number"]), _esc(la["street"]), _esc(la["city"]),
                    _esc(la["state"]), _esc(la["zip"]), _esc(L["legal_description"])))
    o.append('</table>')

    pp = d["pay_plan"]
    o.append('<h2 class="plain">Pay Plan</h2>')
    o.append('<div style="margin-top:5pt">%s%s%s%s</div>'
             % (_kv("Payment Plan:", pp["payment_plan"]),
                _kv("Number of Payments:", pp["number_of_payments"]),
                _kv("Bill Method:", pp["bill_method"]), _kv("Payor:", pp["payor"])))

    # ---------------- final page: disclaimer ----------------
    # Rendered from the same constant that build_sample() puts in the golden JSON,
    # so the page and the ground truth cannot drift apart.
    o.append('<div style="break-before:page"></div>')
    o.append('<div class="disc">%s</div>' % _esc(d["disclaimer"]))

    css = _font_face_css() + _scale_fonts(CSS.replace("__BLUE__", BLUE).replace("__GRAY__", GRAY)
                       .replace("__PRINTED__", meta["printed_on"])
                       .replace("__TIME__", meta["printed_at"]))
    # The DOCTYPE is load-bearing: without it browsers render in quirks mode, where
    # tables do not inherit font-size from body, so every table cell would come out
    # at the 16px default instead of 8.2pt. WeasyPrint does not implement that quirk,
    # which is why the reference renders correctly without one.
    return ("<!DOCTYPE html><html><head><meta charset='utf-8'><style>%s</style>"
            "</head><body>%s</body></html>") % (css, "".join(o))


def render_pdf_bytes(record):
    """Render to PDF twice so `total_pages` in the golden JSON matches reality."""
    meta = record["document_metadata"]
    pdf = html_to_pdf(render_html(record), meta["printed_on"], meta["printed_at"])
    doc = pdfium.PdfDocument(pdf)
    n = len(doc)
    doc.close()
    if n != meta["total_pages"]:
        meta["total_pages"] = n
        pdf = html_to_pdf(render_html(record), meta["printed_on"], meta["printed_at"])
    return pdf


# ===========================================================================
# 4. SCAN SIMULATION
# ===========================================================================
def scanify_page(im, rng, level="none"):
    """Optional scan artefacts: skew, uneven illumination, sensor noise, softness.

    The reference declaration is an image-only PDF but a CLEAN one - its page
    background measures 254/255, where the full degradation below drags it to
    ~239. Grey paper makes black text look highlighted by comparison, so
    'none' is the default: the page is still rasterised (no text layer, so OCR
    is still exercised) but not degraded.
    """
    if level == "none":
        return im

    if level == "light":
        a = np.asarray(im).astype(np.float32)
        a += np.random.normal(0, rng.uniform(0.8, 1.6), a.shape)   # sensor grain only
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        return im.filter(ImageFilter.GaussianBlur(rng.uniform(0.15, 0.3)))

    im = im.rotate(rng.uniform(-0.55, 0.55), resample=Image.BICUBIC,
                   fillcolor=(255, 255, 255), expand=False)
    a = np.asarray(im).astype(np.float32)
    h, w, _ = a.shape
    gy = np.linspace(rng.uniform(0.96, 1.0), rng.uniform(0.93, 0.99), h)[:, None]
    gx = np.linspace(rng.uniform(0.97, 1.0), rng.uniform(0.95, 1.0), w)[None, :]
    a *= (gy * gx)[:, :, None]
    a[:, :, 2] *= 0.985                                   # faint warm paper cast
    a += np.random.normal(0, rng.uniform(2.2, 4.5), a.shape)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.6)))
    return ImageEnhance.Contrast(im).enhance(rng.uniform(1.02, 1.12))


def make_scanned_pdf(clean_pdf_bytes, rng, dpi=200, quality=(72, 88), level="none"):
    """Rasterize -> degrade -> reassemble as an image-only (scanned) PDF."""
    doc = pdfium.PdfDocument(clean_pdf_bytes)
    pages = []
    for i in range(len(doc)):
        im = doc[i].render(scale=dpi / 72.0).to_pil().convert("RGB")
        im = scanify_page(im, rng, level)
        jb = io.BytesIO()
        q = 96 if level == "none" else rng.randint(*quality)
        im.save(jb, "JPEG", quality=q)
        pages.append(Image.open(io.BytesIO(jb.getvalue())).convert("RGB"))
    doc.close()
    out = io.BytesIO()
    pages[0].save(out, "PDF", resolution=float(dpi), save_all=True,
                  append_images=pages[1:])
    return out.getvalue(), pages


# ===========================================================================
# 5. VALIDATION
# ===========================================================================
def pdf_text(pdf_bytes):
    doc = pdfium.PdfDocument(pdf_bytes)
    txt = []
    for i in range(len(doc)):
        tp = doc[i].get_textpage()
        txt.append(tp.get_text_range())
        tp.close()
    doc.close()
    return re.sub(r"\s+", " ", " ".join(txt))


def key_skeleton(o, path=""):
    """Key-shape signature. Array length is ignored; a list of scalars always
    contributes the same marker whether or not it happens to be empty, so an
    empty-but-valid list is not reported as a schema difference."""
    if isinstance(o, dict):
        out = []
        for k in sorted(o):
            out += key_skeleton(o[k], path + "." + k)
        return out
    if isinstance(o, list):
        if o and isinstance(o[0], (dict, list)):
            return key_skeleton(o[0], path + "[]")
        return [path + "[]"]
    return [path]


def validate(record, clean_pdf_bytes, sid):
    """Return a list of problem strings (empty == clean)."""
    problems = []
    flat = pdf_text(clean_pdf_bytes)

    def num(x):
        return float(x.replace("$", "").replace(",", ""))

    # arithmetic
    vc = record["states"][0]["locations"][0]["vehicle_coverages"]
    for ci in range(len(vc["vehicle_columns"])):
        s = sum(num(r["premiums"][ci]) for r in vc["rows"]
                if r["premiums"][ci].startswith("$"))
        if abs(s - num(vc["totals"][ci])) > 0.01:
            problems.append("%s: column %d total mismatch" % (sid, ci + 1))
    grand = sum(num(t) for t in vc["totals"]) + num(record["line_of_business"]["total"])
    if abs(grand - num(record["transaction"]["term_amount"])) > 0.01:
        problems.append("%s: term amount does not equal sum of parts" % sid)

    # groundedness: every scalar must appear in the rendered page text
    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, path + "." + k)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, "%s[%d]" % (path, i))
        elif o is not None and not isinstance(o, int):
            v = str(o).strip()
            if not v or path.endswith(".document_type"):
                return                       # derived label, never printed
            if v in flat:
                return
            pos = 0                          # cells wrap: match word sequence
            for w in v.split():
                i = flat.find(w, pos)
                if i < 0:
                    problems.append("%s: value not found in PDF %s = %r" % (sid, path, o))
                    return
                pos = i + len(w)
    walk(record)
    return problems


# ===========================================================================
# 6. CLI
# ===========================================================================
def parse_args(argv):
    # Support the shorthand count flag: --10 == -n 10
    cleaned, shorthand = [], None
    for a in argv:
        m = re.fullmatch(r"--(\d+)", a)
        if m:
            shorthand = int(m.group(1))
        else:
            cleaned.append(a)

    p = argparse.ArgumentParser(
        prog="generate_policy_dataset.py",
        description="Generate synthetic policy declarations: scanned PDF + golden JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  %(prog)s -i ./New_Business_6.pdf -o ./dataset --10\n"
               "  %(prog)s -o ./out -n 500 --seed 42\n")
    p.add_argument("-i", "--input", default=None,
                   help="Reference PDF (optional; recorded for provenance, not parsed)")
    p.add_argument("-o", "--output", default="./data",
                   help="Base output directory. PDFs go to <output>/training data/, "
                        "golden JSON to <output>/golden json/, each pair sharing one "
                        "file name. Default: ./data")
    p.add_argument("--pdf-dir", default=None,
                   help="Override the PDF directory (default: <output>/training data)")
    p.add_argument("--json-dir", default=None,
                   help="Override the golden JSON directory (default: <output>/golden json)")
    p.add_argument("-n", "--num", type=int, default=None,
                   help="Number of samples. Shorthand: --10, --250, ...")
    p.add_argument("--seed", type=int, default=20260825, help="RNG seed for reproducibility")
    p.add_argument("--dpi", type=int, default=200, help="Scan resolution. Default: 200")
    p.add_argument("--start-index", type=int, default=1, help="First sample number")
    p.add_argument("--carrier", default="GEICO Marine Insurance Co",
                   help="Pin the carrier so every sample is the same declaration. "
                        "Pass '' to cycle through all carriers instead.")
    p.add_argument("--state", default="New York",
                   help="Pin the state (this drives which coverages appear). "
                        "Pass '' to cycle through all states instead.")
    p.add_argument("--vehicles", type=int, default=3,
                   help="Vehicles per sample; matches the reference. 0 = randomise")
    p.add_argument("--drivers", type=int, default=3,
                   help="Drivers per sample; matches the reference. 0 = randomise")
    p.add_argument("--interests", type=int, default=1,
                   help="Additional interests per sample; matches the reference. 0 = randomise")
    p.add_argument("--contacts", type=int, default=1,
                   help="Contact rows per sample; matches the reference. 0 = randomise")
    p.add_argument("--scan", choices=["none", "light", "full"], default="none",
                   help="Scan artefacts to apply. none (default) matches the reference: "
                        "rasterised, image-only, but clean. light adds mild grain. "
                        "full adds skew, uneven illumination and JPEG artefacts.")
    p.add_argument("--no-scan", action="store_true",
                   help="Emit clean digital PDFs WITH a text layer (no rasterisation)")
    p.add_argument("--no-validate", action="store_true", help="Skip validation checks")
    p.add_argument("--quiet", action="store_true", help="Suppress per-sample output")

    args = p.parse_args(cleaned[1:])
    if shorthand is not None:
        if args.num is not None and args.num != shorthand:
            p.error("conflicting counts: --%d and -n %d" % (shorthand, args.num))
        args.num = shorthand
    if args.num is None:
        args.num = 10
    if args.num < 1:
        p.error("sample count must be >= 1")
    return args


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv)

    if args.input and not os.path.isfile(args.input):
        sys.exit("Reference PDF not found: %s" % args.input)

    base = os.path.abspath(args.output)
    pdf_dir = os.path.abspath(args.pdf_dir) if args.pdf_dir else os.path.join(base, "training data")
    json_dir = os.path.abspath(args.json_dir) if args.json_dir else os.path.join(base, "golden json")
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    if not args.quiet:
        print("renderer: %s" % RENDERER)
        print("document : %s / %s / %s vehicles / %s drivers"
              % (args.carrier or "(all carriers)", args.state or "(all states)",
                 args.vehicles or "random", args.drivers or "random"))
        print("PDFs  -> %s" % pdf_dir)
        print("JSON  -> %s" % json_dir)

    problems = []
    width = max(2, len(str(args.start_index + args.num - 1)))

    for k in range(args.num):
        idx = args.start_index + k
        sid = "sample_%0*d" % (width, idx)
        rng = random.Random("%d:%d" % (args.seed, idx))     # per-sample deterministic

        record = build_sample(idx - 1, rng, args.carrier, args.state,
                              args.vehicles, args.drivers, args.interests, args.contacts)
        clean = render_pdf_bytes(record)

        if not args.no_validate:
            problems += validate(record, clean, sid)

        pdf_bytes = (clean if args.no_scan else
                     make_scanned_pdf(clean, rng, dpi=args.dpi, level=args.scan)[0])

        # Two artifacts per sample, in separate directories but sharing one name,
        # so <name>.pdf and <name>.json are obviously a pair.
        with open(os.path.join(pdf_dir, sid + ".pdf"), "wb") as fh:
            fh.write(pdf_bytes)
        with open(os.path.join(json_dir, sid + ".json"), "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, ensure_ascii=False)

        if k == 0:
            ref_keys = key_skeleton(record)
        elif not args.no_validate and key_skeleton(record) != ref_keys:
            problems.append("%s: key set differs from first sample" % sid)

        if not args.quiet:
            print("%s.pdf + %s.json  %-34s %-14s %dv/%dd  %dp  %s"
                  % (sid, sid, record["carrier_name"], record["states"][0]["state"],
                     len(record["vehicles"]), len(record["drivers"]),
                     record["document_metadata"]["total_pages"],
                     record["transaction"]["term_amount"]))

    close_renderer()
    print("\n%d PDF + JSON pairs  [renderer: %s]" % (args.num, RENDERER))
    print("  PDFs: %s" % pdf_dir)
    print("  JSON: %s" % json_dir)
    if args.no_validate:
        print("validation skipped")
    elif problems:
        print("VALIDATION FAILED (%d issues):" % len(problems))
        for pr in problems[:25]:
            print("  -", pr)
        if len(problems) > 25:
            print("  ... and %d more" % (len(problems) - 25))
        return 1
    else:
        print("validation passed: schema consistent, totals tie, all values grounded in PDF")
    return 0


if __name__ == "__main__":
    sys.exit(main())