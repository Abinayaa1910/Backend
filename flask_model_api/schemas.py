from pydantic import BaseModel, Field
from typing import Literal

from pydantic import BaseModel, Field
from typing import Literal
# PromoRequest (Pydantic schema)
# Used in: /generate-promo route
#
# Purpose:
# - Validates incoming JSON from the frontend's prompt generation form
# - Ensures all required campaign and user fields are present and correctly typed
# - Prevents invalid values using Literal enums (e.g. allowed platforms, tones, etc.)

# Why this matters:
# - Keeps the backend safe from malformed data
# - Makes the logic in generate_promo() cleaner (data is already parsed and validated)

class PromoRequest(BaseModel):
    gender: Literal['Male', 'Female', 'Other']
    location: str
    loyalty_tier: Literal['Silver', 'Gold', 'Platinum']
    join_year: int = Field(..., ge=2000, le=2100)
    join_month: Literal[
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    objective: str
    industry: str
    marketing_funnel_stage: Literal['Awareness', 'Consideration', 'Decision', 'Loyalty']
    past_engagement: Literal['High', 'Moderate', 'Low']
    platform: Literal['Instagram', 'Facebook', 'TikTok', 'Email', 'Website', 'LinkedIn']
    post_type: Literal['Text', 'Image', 'Both']
    tone: Literal['Professional', 'Casual', 'Playful', 'Empathetic', 'Fun']
    num_variants: Literal[1, 2, 3]
