import re


def _normalize_whitespace(text: str) -> str:
    cleaned = re.sub(r"^\s*(?:\d+(?:\.\d+)*|[A-Za-z]\)|\([a-z]\))\s*", "", (text or ""))
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _extract_relevant_sentences(context: str, question: str) -> list[str]:
    if not context:
        return []

    candidates = re.split(r"(?<=[.?!\n])\s+", context)
    question_terms = [
        term
        for term in re.findall(r"[a-zA-Z][a-zA-Z\-']+", (question or "").lower())
        if len(term) > 2
    ]

    keywords = {
        "scheme",
        "schemes",
        "beneficiary",
        "beneficiaries",
        "farmer",
        "farmers",
        "eligible",
        "eligibility",
        "support",
        "subsidy",
        "insurance",
        "loan",
        "kisan",
        "pm-kisan",
        "pmkisan",
        "pmfby",
        "kcc",
    }

    if question_terms:
        keywords.update(question_terms)

    scored: list[tuple[int, str]] = []
    for sentence in candidates:
        cleaned = _normalize_whitespace(sentence)
        if not cleaned:
            continue
        lower = cleaned.lower()
        score = sum(1 for keyword in keywords if keyword in lower)
        if score:
            scored.append((score, cleaned))

    scored.sort(key=lambda item: item[0], reverse=True)

    seen = set()
    results = []
    for _, sentence in scored:
        if sentence not in seen:
            seen.add(sentence)
            results.append(sentence)
        if len(results) >= 4:
            break

    return results


def _question_intent(question: str) -> str:
    lowered = (question or "").lower()

    if any(word in lowered for word in ("apply", "application", "how can i apply", "how do i apply", "registration", "enroll", "enrol")):
        return "apply"
    if any(word in lowered for word in ("eligible", "eligibility", "who can", "who is")):
        return "eligibility"
    if any(word in lowered for word in ("benefit", "benefits", "what is", "about", "scheme")):
        return "benefit"
    if any(word in lowered for word in ("claim", "claims", "how to claim", "settle")):
        return "claim"
    if any(word in lowered for word in ("premium", "cost", "price", "fee")):
        return "premium"
    if any(word in lowered for word in ("document", "documents", "paper", "required")):
        return "documents"
    return "general"


def _pick_intent_sentences(sentences: list[str], intent: str) -> list[str]:
    intent_keywords = {
        "apply": (
            "apply",
            "application",
            "submit",
            "portal",
            "app",
            "form",
            "acknowledgement",
            "insurance company",
            "bank",
            "intermediary",
            "premium",
        ),
        "eligibility": (
            "eligible",
            "eligibility",
            "beneficiary",
            "beneficiaries",
            "farmer",
            "farmers",
            "landholding",
            "cultivable",
            "state",
            "union territory",
        ),
        "benefit": (
            "benefit",
            "scheme",
            "support",
            "insurance",
            "credit",
            "loan",
            "income support",
            "coverage",
        ),
        "claim": (
            "claim",
            "claims",
            "loss",
            "settle",
            "assessment",
            "compensation",
        ),
        "premium": (
            "premium",
            "subsidy",
            "share",
            "cost",
            "payment",
        ),
        "documents": (
            "document",
            "documents",
            "proof",
            "certificate",
            "identity",
            "registration",
            "bank",
        ),
    }

    keywords = intent_keywords.get(intent, ())
    if not keywords:
        return sentences[:3]

    def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
        lower = text.lower()
        return any(keyword in lower for keyword in keywords)

    picked = [sentence for sentence in sentences if _contains_any(sentence, keywords)]
    if picked:
        return picked[:3]

    return sentences[:3]


def _build_direct_answer(question: str, context: str) -> str | None:
    normalized_question = (question or "").lower()
    sentences = _extract_relevant_sentences(context, question)
    intent = _question_intent(question)

    if not sentences:
        return None

         # ==================================================
    # CSV Scheme Detection
    # ==================================================

    scheme_match = re.search(
        r"scheme_name:\s*(.+?)\n.*?details:\s*(.+)",
        context,
        re.IGNORECASE | re.DOTALL
    )

    if scheme_match:

        scheme_name = scheme_match.group(1).strip()
        details = scheme_match.group(2).strip()

        return (
            f"✅ <b>{scheme_name}</b><br><br>"
            f"{details[:1000]}"
        )
    
    def _pick_points(keywords: tuple[str, ...], labels: tuple[str, ...]) -> list[str]:
        picked = []
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                picked.append(sentence)
        if picked:
            return picked[:3]
        return [label for label in labels]

    def _format_points(points: list[str]) -> str:
        formatted = []
        seen = set()
        for point in points:
            clean_point = _normalize_whitespace(point)
            if clean_point and clean_point not in seen:
                seen.add(clean_point)
                formatted.append(f"✅ {clean_point}")
        return "<br>".join(formatted)

    scheme_aliases = {
        "PM-KISAN": ("pm-kisan", "pm kisan", "pradhan mantri kisan samman nidhi", "kisan samman nidhi"),
        "PMFBY": ("pmfby", "pm fby", "pradhan mantri fasal bima yojana", "crop insurance"),
        "KCC": ("kcc", "kisan credit card", "credit card"),
        "RKVY": ("rkvy", "rashtriya krishi vikas yojana"),
    }

    detected_scheme = None
    question_or_context = f"{question}\n{context}".lower()
    for scheme_name, aliases in scheme_aliases.items():
        if any(alias in question_or_context for alias in aliases):
            detected_scheme = scheme_name
            break

    if detected_scheme == "PM-KISAN" or "pm-kisan" in normalized_question or "pm kisan" in normalized_question:
        summary_points = [
            "PM-KISAN provides income support to landholding farmer families with cultivable land.",
            "Eligible beneficiaries are identified by the State or Union Territory.",
            "Non-resident Indians (NRIs) are excluded from the scheme.",
        ]

        if any("6000" in sentence.lower() or "rs.6000" in sentence.lower() or "6000/-" in sentence.lower() for sentence in sentences):
            summary_points.append(
                "The scheme releases Rs. 6000 per year to eligible farmers through Direct Benefit Transfer."
            )

        if intent == "apply":
            application_points = _pick_intent_sentences(
                sentences,
                "apply",
            )
            if application_points:
                return _format_points(application_points)

        return _format_points(summary_points[:4])

    if detected_scheme == "PMFBY":
        if intent == "apply":
            application_points = _pick_intent_sentences(sentences, "apply")
            return _format_points(application_points)

        if intent == "claim":
            claim_points = _pick_intent_sentences(sentences, "claim")
            return _format_points(claim_points)

        if intent == "premium":
            premium_points = _pick_intent_sentences(sentences, "premium")
            return _format_points(premium_points)

        summary_points = [
            "PMFBY is a crop insurance scheme for farmers.",
            "It helps protect against crop loss from natural calamities, pests, and diseases.",
            "Claims are settled based on the crop loss assessment and scheme rules.",
        ]
        return _format_points(summary_points[:3])

    if detected_scheme == "KCC":
        if intent == "apply":
            application_points = _pick_intent_sentences(sentences, "apply")
            if application_points:
                return _format_points(application_points)

            return _format_points(
                [
                    "Visit your bank or lending institution to apply for Kisan Credit Card.",
                    "Submit the required land, identity, and banking documents as per bank guidelines.",
                    "The bank will process the application and sanction credit based on eligibility.",
                ]
            )

        if intent in {"benefit", "general"}:
            return _format_points(
                [
                    "Kisan Credit Card provides short-term agricultural credit to farmers.",
                    "It helps farmers get timely and flexible loans for farming and allied needs.",
                    "It supports crop production expenses and related activities.",
                ]
            )

        if intent == "eligibility":
            eligibility_points = _pick_intent_sentences(sentences, "eligibility")
            if eligibility_points:
                return _format_points(eligibility_points)
            return _format_points(
                [
                    "KCC is available to eligible farmers and related agricultural beneficiaries.",
                    "Banks decide the final eligibility based on scheme and lending rules.",
                ]
            )

        return _format_points(
            [
                "Kisan Credit Card provides short-term agricultural credit to farmers.",
                "It helps farmers get timely and flexible loans for farming and allied needs.",
                "It supports crop production expenses and related activities.",
            ]
        )

    if detected_scheme == "RKVY":
        if intent == "benefit":
            benefit_points = _pick_intent_sentences(sentences, "benefit")
            return _format_points(benefit_points)

        summary_points = [
            "RKVY supports agricultural development and productivity growth.",
            "It helps states strengthen farming-related projects and infrastructure.",
            "It encourages innovation in agriculture and allied sectors.",
        ]
        return _format_points(summary_points[:3])

    if any(alias in f"{question_or_context}" for aliases in scheme_aliases.values() for alias in aliases):
        selected_points = _pick_intent_sentences(sentences, intent)
        if selected_points:
            return _format_points(selected_points)

    if any(word in normalized_question for word in ("eligible", "eligibility", "who can", "who is", "apply")):
        if any("nri" in sentence.lower() for sentence in sentences):
            return (
                "✅ PM-KISAN is for landholding farmer families with cultivable land.<br>"
                "✅ Eligible beneficiaries are identified by the State or Union Territory.<br>"
                "✅ NRIs are excluded from the scheme."
            )
    
    if len(sentences) == 1:
        return _format_points([sentences[0]])

    return _format_points(sentences[:3])


def generate_answer(question: str, context: str) -> str:
    """
    Generate answer using retrieved context only.
    No Gemini, no OpenAI, no LLM.
    """

    def _extract_evidence(ctx: str):
        if not ctx:
            return []

        candidates = re.split(r"(?<=[.\n])\s+", ctx)

        keywords = [
            "eligible",
            "eligibility",
            "scheme",
            "beneficiary",
            "beneficiaries",
            "farmers",
            "farmer",
            "support",
            "subsidy",
            "insurance",
            "loan",
        ]

        found = []

        for sentence in candidates:
            lower = sentence.lower()

            if any(keyword in lower for keyword in keywords):
                cleaned = sentence.strip()

                if cleaned and cleaned not in found:
                    found.append(cleaned)

        return found

    # First use your existing rule-based NLP logic
    direct_answer = _build_direct_answer(question, context)

    if direct_answer:
        return direct_answer

    # Fallback: extract useful evidence sentences
    evidence = _extract_evidence(context)

    if evidence:
        return "<br>".join(
            f"✅ {_normalize_whitespace(sentence)}"
            for sentence in evidence[:3]
        )

    return "Information not found in the retrieved scheme documents."

def recommend(profile: dict) -> list:
    schemes = []
    crop = (profile.get("crop") or "").lower()

    if "cotton" in crop:
        schemes = ["PMFBY", "PM-KISAN", "KCC"]
    elif "wheat" in crop:
        schemes = ["PM-KISAN", "RKVY", "KCC"]
    elif "rice" in crop:
        schemes = ["PMFBY", "PM-KISAN", "KCC"]
    else:
        schemes = ["PM-KISAN", "KCC"]

    return schemes
    