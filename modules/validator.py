# ============================================================
# modules/validator.py — Rule-Based Validation Engine (v2)
#
# UPDATED for multi-property-type support (Residential, Flat,
# Commercial, IT_Park, Vacant_Site, Industrial).
#
# Key change from v1: rules are now split into
#   - UNIVERSAL_RULES   -> run for every property type
#   - TYPE_RULES        -> only run for the matching property_type
#
# "Not Applicable" is treated as an automatic PASS (never flagged).
# "Not found" on a field that DOES apply is still flagged, same as
# before.
#
# This still is plain Python logic — NOT an AI call. Deterministic,
# fast, free, and always gives the same result for the same input.
# ============================================================


def _is_missing(value):
    # Helper — treats None, empty string, "Not found", "N/A"
    # and similar phrasings as "missing"
    if value is None:
        return True
    text = str(value).strip().lower()
    return text in (
        "", "not found", "n/a", "na", "not mentioned",
        "not explicitly mentioned", "unknown", "none"
    )


def _is_not_applicable(value):
    # NEW — distinguishes "doesn't apply to this property type"
    # from "should be there but is missing"
    if value is None:
        return False
    return str(value).strip().lower() in ("not applicable", "n.a.", "na - not applicable")


def _get_nested(data, section, field):
    # Helper for reading fields nested under residential_fields,
    # flat_fields, commercial_fields, it_park_fields,
    # vacant_site_fields, industrial_fields
    section_data = data.get(section, {})
    if not isinstance(section_data, dict):
        return None
    return section_data.get(field)


# --------------------------------------------------------
# UNIVERSAL RULES — apply to every property type
# --------------------------------------------------------

def check_property_type(data):
    if _is_missing(data.get("property_type")):
        return {
            "level": "CRITICAL",
            "field": "Property Type",
            "message": "Property type could not be determined from the document."
        }
    return None


def check_owner_name(data):
    if _is_missing(data.get("owner_name")):
        return {
            "level": "CRITICAL",
            "field": "Owner Name",
            "message": "Owner name is missing. Cannot proceed without owner identification."
        }
    return None


def check_property_address(data):
    if _is_missing(data.get("property_address")):
        return {
            "level": "CRITICAL",
            "field": "Property Address",
            "message": "Property address / schedule is missing."
        }
    return None


def check_survey_number(data):
    if _is_missing(data.get("survey_number")):
        return {
            "level": "CRITICAL",
            "field": "Survey Number",
            "message": "Survey number is missing. Required for SRO verification."
        }
    return None


def check_total_area(data):
    area = data.get("total_area_sqft")
    if _is_missing(area):
        return {
            "level": "CRITICAL",
            "field": "Total Area",
            "message": "Total area is missing from the document."
        }
    try:
        cleaned = (
            str(area).lower()
            .replace("sq ft", "").replace("sqft", "")
            .replace("sq.ft", "").replace(",", "").strip()
        )
        numeric_area = float(cleaned)
        if numeric_area <= 0 or numeric_area > 200000:
            return {
                "level": "WARNING",
                "field": "Total Area",
                "message": (
                    f"Area value '{area}' looks unusual — "
                    f"please verify manually."
                )
            }
    except (ValueError, TypeError):
        return {
            "level": "WARNING",
            "field": "Total Area",
            "message": (
                f"Area value '{area}' could not be parsed as a number — "
                f"please verify manually."
            )
        }
    return None


def check_market_value(data):
    if _is_missing(data.get("market_value")):
        return {
            "level": "CRITICAL",
            "field": "Market Value",
            "message": "Market value is missing — required for loan decisioning."
        }
    return None


def check_registration_date(data):
    if _is_missing(data.get("registration_date")):
        return {
            "level": "WARNING",
            "field": "Registration Date",
            "message": "Registration date not found — verify against SRO records."
        }
    return None


def check_litigation_status(data):
    value = str(data.get("litigation_status", "")).lower()
    if "pending" in value or "dispute" in value or "case" in value:
        return {
            "level": "CRITICAL",
            "field": "Litigation Status",
            "message": f"Possible litigation flagged: '{data.get('litigation_status')}'"
        }
    if _is_missing(data.get("litigation_status")):
        return {
            "level": "INFO",
            "field": "Litigation Status",
            "message": "Litigation status not explicitly confirmed in document."
        }
    return None


def check_encumbrance_status(data):
    value = str(data.get("encumbrance_status", "")).lower()
    if "not clear" in value or ("encumbrance" in value and "no encumbrance" not in value):
        return {
            "level": "WARNING",
            "field": "Encumbrance Status",
            "message": f"Encumbrance flag detected: '{data.get('encumbrance_status')}'"
        }
    if _is_missing(data.get("encumbrance_status")):
        return {
            "level": "INFO",
            "field": "Encumbrance Status",
            "message": "Encumbrance Certificate (EC) status not found in document."
        }
    return None


def check_guideline_value(data):
    if _is_missing(data.get("guideline_value")):
        return {
            "level": "INFO",
            "field": "Guideline Value",
            "message": "Government guideline value not found — nice to have for comparison."
        }
    return None


UNIVERSAL_RULES = [
    check_property_type,
    check_owner_name,
    check_property_address,
    check_survey_number,
    check_total_area,
    check_market_value,
    check_registration_date,
    check_litigation_status,
    check_encumbrance_status,
    check_guideline_value,
]


# --------------------------------------------------------
# TYPE-SPECIFIC RULES — only run when property_type matches
# --------------------------------------------------------

def _residential_rules(data):
    flags = []
    section = "residential_fields"

    building_age = _get_nested(data, section, "building_age_years")
    if not _is_not_applicable(building_age) and _is_missing(building_age):
        flags.append({
            "level": "WARNING",
            "field": "Building Age",
            "message": "Building age not found — needed for depreciation calculation."
        })

    boundary = _get_nested(data, section, "boundary_compliance")
    if not _is_not_applicable(boundary) and _is_missing(boundary):
        flags.append({
            "level": "WARNING",
            "field": "Boundary Compliance",
            "message": "Boundary compliance status not mentioned — verify manually."
        })

    return flags


def _flat_rules(data):
    flags = []
    section = "flat_fields"

    for field, label, level in [
        ("flat_number", "Flat Number", "CRITICAL"),
        ("society_apartment_name", "Society/Apartment Name", "WARNING"),
        ("undivided_share_of_land_sqft", "Undivided Share (UDS)", "WARNING"),
        ("occupancy_certificate_status", "Occupancy Certificate", "CRITICAL"),
    ]:
        value = _get_nested(data, section, field)
        if not _is_not_applicable(value) and _is_missing(value):
            flags.append({
                "level": level,
                "field": label,
                "message": f"{label} is missing — required for flat/apartment loans."
            })

    return flags


def _commercial_rules(data):
    flags = []
    section = "commercial_fields"

    usage = _get_nested(data, section, "usage_permission_type")
    if not _is_not_applicable(usage) and _is_missing(usage):
        flags.append({
            "level": "CRITICAL",
            "field": "Usage Permission",
            "message": "Commercial usage permission type not found — required to confirm legal use."
        })

    occ_cert = _get_nested(data, section, "commercial_occupancy_certificate")
    if not _is_not_applicable(occ_cert) and _is_missing(occ_cert):
        flags.append({
            "level": "CRITICAL",
            "field": "Occupancy Certificate",
            "message": "Commercial occupancy certificate status not found."
        })

    fire_noc = _get_nested(data, section, "fire_safety_noc_status")
    if not _is_not_applicable(fire_noc) and _is_missing(fire_noc):
        flags.append({
            "level": "WARNING",
            "field": "Fire Safety NOC",
            "message": "Fire safety NOC status not found — verify before bank submission."
        })

    return flags


def _it_park_rules(data):
    flags = []
    section = "it_park_fields"

    sez = _get_nested(data, section, "sez_status")
    if not _is_not_applicable(sez) and _is_missing(sez):
        flags.append({
            "level": "WARNING",
            "field": "SEZ Status",
            "message": "SEZ status not confirmed — affects tax and resale treatment."
        })

    cert = _get_nested(data, section, "it_park_certification_status")
    if not _is_not_applicable(cert) and _is_missing(cert):
        flags.append({
            "level": "CRITICAL",
            "field": "IT Park Certification",
            "message": "IT Park / STPI certification status not found."
        })

    return flags


def _vacant_site_rules(data):
    flags = []
    section = "vacant_site_fields"

    zoning = _get_nested(data, section, "zoning_classification")
    if not _is_not_applicable(zoning) and _is_missing(zoning):
        flags.append({
            "level": "CRITICAL",
            "field": "Zoning Classification",
            "message": "Zoning classification not found — required to confirm permitted land use."
        })

    conversion = _get_nested(data, section, "conversion_status")
    if not _is_not_applicable(conversion) and _is_missing(conversion):
        flags.append({
            "level": "WARNING",
            "field": "Land Conversion Status",
            "message": "Agricultural-to-non-agricultural conversion status not found — verify if applicable."
        })

    return flags


def _industrial_rules(data):
    flags = []
    section = "industrial_fields"

    factory_license = _get_nested(data, section, "factory_license_status")
    if not _is_not_applicable(factory_license) and _is_missing(factory_license):
        flags.append({
            "level": "CRITICAL",
            "field": "Factory License",
            "message": "Factory license status not found — required for industrial property loans."
        })

    pollution = _get_nested(data, section, "pollution_control_clearance")
    if not _is_not_applicable(pollution) and _is_missing(pollution):
        flags.append({
            "level": "CRITICAL",
            "field": "Pollution Control Clearance",
            "message": "Pollution control clearance not found — mandatory for industrial units."
        })

    zoning = _get_nested(data, section, "zoning_classification")
    if not _is_not_applicable(zoning) and _is_missing(zoning):
        flags.append({
            "level": "WARNING",
            "field": "Zoning Classification",
            "message": "Zoning classification not found — verify industrial land use permission."
        })

    return flags


TYPE_RULES = {
    "Residential": _residential_rules,
    "Flat": _flat_rules,
    "Commercial": _commercial_rules,
    "IT_Park": _it_park_rules,
    "Vacant_Site": _vacant_site_rules,
    "Industrial": _industrial_rules,
}


# --------------------------------------------------------
# MAIN VALIDATION FUNCTION
# --------------------------------------------------------

def validate_property_data(data: dict) -> dict:
    """
    Run all validation rules against one property's extracted data.
    Universal rules always run. Type-specific rules run ONLY for
    the section matching data["property_type"].

    Output: dict with:
            "critical_flags": list of CRITICAL issues
            "warning_flags":  list of WARNING issues
            "info_flags":     list of INFO notes
            "is_bank_ready":  True only if zero CRITICAL flags
            "total_flags":    count of all flags combined
            "property_type":  echoed back for downstream agents
    """
    critical_flags = []
    warning_flags = []
    info_flags = []

    # Universal rules — every property type
    for rule in UNIVERSAL_RULES:
        result = rule(data)
        if result is None:
            continue
        if result["level"] == "CRITICAL":
            critical_flags.append(result)
        elif result["level"] == "WARNING":
            warning_flags.append(result)
        else:
            info_flags.append(result)

    # Type-specific rules — only the matching type's function runs
    property_type = str(data.get("property_type", "")).strip()
    type_rule_fn = TYPE_RULES.get(property_type)

    if type_rule_fn:
        for result in type_rule_fn(data):
            if result["level"] == "CRITICAL":
                critical_flags.append(result)
            elif result["level"] == "WARNING":
                warning_flags.append(result)
            else:
                info_flags.append(result)
    elif property_type:
        # property_type was extracted but doesn't match any known
        # type — flag it so a human checks, rather than silently
        # skipping all type-specific validation
        warning_flags.append({
            "level": "WARNING",
            "field": "Property Type",
            "message": (
                f"Property type '{property_type}' does not match any "
                f"known category — type-specific checks were skipped. "
                f"Verify manually."
            )
        })

    return {
        "critical_flags": critical_flags,
        "warning_flags": warning_flags,
        "info_flags": info_flags,
        "is_bank_ready": len(critical_flags) == 0,
        "total_flags": (
            len(critical_flags) + len(warning_flags) + len(info_flags)
        ),
        "property_type": property_type or "Unknown",
    }


def validate_merged_property(per_document_data: list) -> dict:
    """
    For merged property cases — validates EACH document separately,
    then checks one cross-document rule.
    (Unchanged from v1 — merge logic is independent of property type.)
    """
    per_document_results = []
    for doc_data in per_document_data:
        result = validate_property_data(doc_data)
        result["document_name"] = doc_data.get("document_name", "Unknown")
        per_document_results.append(result)

    cross_document_notes = []

    owner_names = [
        d.get("owner_name") for d in per_document_data
        if not _is_missing(d.get("owner_name"))
    ]
    if len(set(owner_names)) <= 1 and len(owner_names) > 1:
        cross_document_notes.append({
            "level": "WARNING",
            "field": "Owner Names",
            "message": (
                "All documents show the same owner name — "
                "expected different owners for a merged property. "
                "Please verify this is genuinely a merge case."
            )
        })

    return {
        "per_document": per_document_results,
        "cross_document_notes": cross_document_notes
    }