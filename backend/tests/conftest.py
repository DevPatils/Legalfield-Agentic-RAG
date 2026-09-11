"""Shared fixtures: a synthetic contract exercising the numbering conventions we parse."""

import pytest

SAMPLE_CONTRACT = """\
MUTUAL NON-DISCLOSURE AGREEMENT

This Agreement is entered into as of January 1, 2024 (the "Effective Date").

ARTICLE I - DEFINITIONS

1.1 Defined Terms. Capitalized terms have the meanings given in this Article I.

1.3 Confidential Information. "Confidential Information" means any non-public
information disclosed by one party (the "Disclosing Party") to the other party
(the "Receiving Party"), whether orally or in writing.

ARTICLE IV - CONFIDENTIALITY

Section 4.1 Obligations. The Receiving Party shall protect all Confidential
Information with the same degree of care it uses for its own information.

Section 4.2 Exceptions. Notwithstanding Section 4.1, the obligations set forth in
Section 4.1 shall not apply to information that:
(a) is publicly available through no fault of the Receiving Party;
(b) was known to the Receiving Party prior to disclosure, provided that:
(i) such knowledge is documented in writing; and
(ii) the documentation predates the Effective Date;
(c) is independently developed as described in Sections 4.3 and 4.4.

Section 4.3 Independent Development. Development without reference to Confidential
Information is permitted.

Section 4.4 Residuals. Nothing herein restricts the use of residual knowledge.

ARTICLE VIII - TERM AND TERMINATION

8.1 Term. This Agreement commences on the Effective Date.

8.2 Termination. Either party may terminate this Agreement upon thirty (30) days
written notice, subject to Section 8.3 and pursuant to Article IV.

8.3 Survival. The obligations in Sections 4.1 through 4.3 survive termination.
"""


@pytest.fixture
def sample_contract() -> str:
    return SAMPLE_CONTRACT


@pytest.fixture
def doc_id() -> str:
    return "doc_001"
