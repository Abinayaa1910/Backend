import numpy as np
from hdbscan.prediction import approximate_predict
from openai import OpenAI
import openai

def get_cluster_label(user_input, clusterer, encoder, scaler, umap_model):
    cat_input = [[
        user_input['location'],
        user_input['gender'],
        int(user_input['join_year']),
        int(user_input['join_month']),
        int(user_input['join_quarter'])
    ]]
    loyalty_score = [[
        {'Silver': 1, 'Gold': 2, 'Platinum': 3}.get(user_input['loyalty_tier'], 1)
    ]]
    encoded = encoder.transform(cat_input)
    scaled = scaler.transform(loyalty_score)
    combined = np.hstack([encoded, scaled])
    embedding = umap_model.transform(combined)

    cluster_id, _ = approximate_predict(clusterer, embedding)
    return cluster_id[0]


def get_openai_response(prompt, api_key):
    openai.api_key = api_key  # Set your API key
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a creative and helpful marketing assistant. "
                    "Your task is to write short, engaging, and personalized promotional messages "
                    "based on the user's profile and cluster persona. Keep it fun, exclusive, and audience-appropriate."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.75,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content.strip()


# Enhanced Cluster-Aware Prompt Generation Logic
def generate_prompt(user_input, clusterer, encoder, scaler, umap_model, cluster_personas, api_key, override_persona=None):
    image_urls = None

    if override_persona:
        persona = override_persona
        cluster_id = None  # Skipping cluster prediction for persona-only generation
    else:
        cluster_id = get_cluster_label(user_input, clusterer, encoder, scaler, umap_model)
        persona = cluster_personas.get(cluster_id, {})


    used_fields = {"from_cluster": [], "from_user_input": [], "optional_inputs": []}
    print(f"[DEBUG] Cluster {cluster_id} Persona Used:", persona)

    user_year = user_input.get("join_year")
    user_tier = user_input.get("loyalty_tier")
    join_month = user_input.get("join_month")
    join_quarter = user_input.get("join_quarter")
    platform = user_input.get("platform", "Instagram")
    post_type = user_input.get("post_type", "Text")
    tone = user_input.get("tone", "Friendly")
    num_variants = int(user_input.get("num_variants", 1))

    persona_years = persona.get("Top_Join_Years", [])
    persona_months = persona.get("Top_Join_Months", [])
    persona_quarter = persona.get("Top_Join_Quarter")

    month_map = {
        1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
        7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
    }

    join_context = ""
    if user_year and persona_years:
        min_year = min(persona_years)
        max_year = max(persona_years)
        if user_year < min_year:
            join_context = f"You've been with us since {user_year}—before many joined in {min_year}–{max_year}. That deserves recognition."
        elif user_year in persona_years:
            join_context = f"You're part of our vibrant community that came aboard in {user_year}!"
        else:
            join_context = f"Thank you for being part of our growing family since {user_year}."

    month_phrase = ""
    if join_month:
        month_name = month_map.get(join_month, "a great month")
        if join_month in persona_months:
            month_phrase = f"You joined in {month_name}, right in the heart of our energetic Q{persona_quarter} wave — a peak time when many came aboard! 🌟"
        else:
            month_phrase = f"Back in {month_name}, you came aboard just after our big Q{persona_quarter} wave — and helped us keep that momentum going strong."
    # Join quarter mention
    quarter_context = ""
    if join_quarter and user_year:
        quarter_context = f"You joined us in Q{join_quarter} of {user_year}, bringing great energy into our community during that time."

    # Compose location blend
    user_location = user_input.get("location")
    persona_locations = persona.get("Top_Locations", [])
    location_phrase = ""
    if user_location and persona_locations:
        if user_location in persona_locations:
            location_phrase = f"You're one of many from {user_location} who’ve shaped our community — that spirit sails strong! 🌊"
        else:
            persona_top = ', '.join(persona_locations[:2])
            location_phrase = f"Whether you're sailing from {user_location} or our top hubs like {persona_top}, you're part of something special."

    # Loyalty tier mention (indirectly)
    tier_context = f"As a valued {user_tier} Tier member, you're at the heart of our journey." if user_tier else ""

    prompt = f"""You are a creative marketing copywriter.

Your job is to generate a short, engaging {user_input.get('industry', 'promotion')} promotion message.

 Marketing Objective: {user_input.get("objective", "Brand Awareness")}
 Industry: {user_input.get("industry", "General")}
 Platform: {platform}
 Post Type: {post_type}
 Tone Preference: {tone}

 Membership Context:
{join_context}
{month_phrase}
{quarter_context}

 Location Insight:
{location_phrase}

Loyalty Context:
{tier_context}

Style Guide:
Take reference from the user's tone and post type above. Emulate the cluster persona’s preferred writing style and interests subtly where suitable. Do not copy exact traits or contradict user data.

Avoid stating exact gender directly. Do not mention persona locations unless naturally relevant.

Now write the message using a natural, friendly tone. Include emojis where appropriate."""

    if user_input.get("post_type") in ["Image", "Both"]:
        try:
            slogan = generate_slogan(user_input, api_key)
            image_prompt = build_image_prompt(user_input, slogan)
            print("🖼️ Final Image Prompt:\n", image_prompt)
            print(f"[DEBUG] Generating {num_variants} image(s) for platform {platform}")
            image_urls = generate_image_content(
                image_prompt, api_key, platform, num_variants=num_variants
            )

            print(f"[DEBUG] Image URLs returned: {image_urls}")
        except Exception as e:
            print(f"❌ Image generation failed: {e}")
            image_urls = None


    used_fields["from_cluster"].extend(["Writing_Style", "Interests", "Special_Offer", "Top_Join_Years", "Top_Join_Months", "Top_Join_Quarter", "Top_Locations"])
    used_fields["from_user_input"].extend(["join_year", "join_month", "join_quarter", "loyalty_tier", "location", "objective", "industry", "platform", "post_type", "tone"])

    result = []

    if api_key != "SKIP_TEXT" and user_input.get("post_type") in ["Text", "Both"]:
        for _ in range(num_variants):
            result.append(get_openai_response(prompt, api_key))
        print(f"✅ Generated {len(result)} text variants")
    else:
        print("🖼️ Image-only mode — no text variants generated")

    return prompt, result, used_fields, image_urls

def generate_prompt_from_persona(persona_summary, persona, api_key,
                                  objective="", industry="", funnel_stage="", past_engagement=""):

    prompt = f"""You are a professional marketing copywriter.

Write a short promotional social media post that follows this internal structure:
- Start with a hook that grabs attention immediately
- Follow with an engaging, benefit-driven message that builds excitement
- End with a clear and persuasive call-to-action

 Target Audience:
{persona_summary}

 Campaign Context:
- Objective: {objective or "Brand Awareness"}
- Industry: {industry or "General"}
- Funnel Stage: {funnel_stage or "Unspecified"}
- Past Engagement Level: {past_engagement or "Unknown"}
 Tone & Style Guidelines:
- Use a friendly and natural tone appropriate for social media
- Include emojis for emotional appeal, but don’t overuse them
- Do NOT include section headers (e.g., "Hook:", "CTA:") — Format the message using **natural paragraph spacing** (i.e., short line breaks between thoughts). Do not label sections. Just write the post with clear, spaced-out paragraphs like a real social media caption.
- Do NOT repeat the persona summary verbatim
- Keep it concise, authentic, and emotionally compelling

Now write the promotional post accordingly.
"""
    print(" Final prompt:\n", prompt)

    result = get_openai_response(prompt, api_key)
    return prompt, result


from openai import OpenAI
import base64

# -- Prompt builder function --
def build_image_prompt(user_input, slogan):
    return (
        "You are a professional Canva-style designer creating a **photo-realistic digital poster** for a marketing campaign.\n\n"
        f"🎯 Objective: {user_input.get('objective', 'Drive awareness')}\n"
        f"🏢 Industry: {user_input.get('industry', 'General')}\n"
        f"📍 Target Audience: {user_input.get('gender', 'All')} users in {user_input.get('location', 'Anywhere')} "
        f"(Loyalty Tier: {user_input.get('loyalty_tier', 'Standard')})\n"
        f"📱 Platform: {user_input.get('platform', 'Instagram')}\n"
        f"🎨 Tone & Style: Clean, realistic, elegant, and minimal\n\n"
        "✅ Design Guidelines:\n"
        "- Use the entire poster space efficiently — do **not** show posters in fake desk settings or 3D mockups.\n"
        "- Background must be clean and relevant to the campaign theme.\n"
        "- The slogan must be placed **clearly** at the top as the only main heading.\n"
        "- Include **no more than 1 line of additional text**, or none at all.\n"
        "- Avoid placing UI elements, emojis, QR codes, or clutter.\n"
        "- Use **real, legible fonts** — no symbols, warped text, or random letters.\n\n"
        "🚫 Avoid:\n"
        "- No fake UI mockups, emojis, or gibberish\n"
        "- No camera overlays or clipped text\n\n"
        f"📝 Only include this heading clearly:\n"
        f"\"{slogan}\"\n"
        "- Use only this slogan. No additional headings or taglines."
    )

# -- Image generation function using GPT-4o + tools --
def generate_image_content(image_prompt, api_key, platform="Instagram", custom_width=None, custom_height=None, num_variants=1):
    import openai
    openai.api_key = api_key

    size_map = {
        "Instagram": "1024x1024",
        "Facebook": "1024x1024",
        "LinkedIn": "1024x1024",
        "TikTok": "1024x1792",        # portrait
        "Email": "1024x1024",
        "Website": "1792x1024",       # landscape
        "Twitter": "1792x1024",       # landscape
        "Pinterest": "1024x1792"      # tall
    }

    if platform == "Custom" and custom_width and custom_height:
        image_size = f"{custom_width}x{custom_height}"
    else:
        image_size = size_map.get(platform, "1024x1024")

    images = []
    for _ in range(num_variants):
        try:
            response = openai.images.generate(
                model="dall-e-3",  
                prompt=image_prompt,
                n=1,
                size=image_size
            )
            images.append(response.data[0].url)
        except Exception as e:
            print(f"❌ Image generation failed for one variant: {e}")

    return images

def generate_slogan(user_input, api_key):
    openai.api_key = api_key
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You're a creative copywriter. Generate one short, catchy slogan (max 7 words) for a marketing campaign."},
            {"role": "user", "content": f"""Objective: {user_input.get('objective', 'Brand Awareness')}
Industry: {user_input.get('industry', 'General')}
Tone: {user_input.get('tone', 'Friendly')}
Platform: {user_input.get('platform', 'Instagram')}"""}
        ],
        temperature=0.8,
        max_tokens=30,
    )
    return response.choices[0].message.content.strip().strip('"')

def build_image_prompt(user_input, slogan):
    return (
        "You are a professional Canva-style designer creating a **photo-realistic digital poster** for a marketing campaign.\n\n"
        
        f"Objective: {user_input.get('objective', 'Drive awareness')}\n"
        f"Industry: {user_input.get('industry', 'General')}\n"
        f"Target Audience: {user_input.get('gender', 'All')} users in {user_input.get('location', 'Anywhere')} "
        f"(Loyalty Tier: {user_input.get('loyalty_tier', 'Standard')})\n"
        f"Platform: {user_input.get('platform', 'Instagram')}\n"
        f"Tone & Style: Clean, realistic, elegant, and minimal\n\n"

        " Design Guidelines:\n"
        "- Use the entire poster space efficiently — do **not** show posters in fake desk settings or 3D mockups.\n"
        "- Background must be clean and relevant to the campaign theme (e.g. resort, travel, luxury, spa, cruise, etc.).\n"
        "- The slogan must be placed **clearly** at the top as the only main heading.\n"
        "- Include **no more than 1 line of additional text**, or none at all.\n"
        "- Avoid placing any other UI elements, social media icons, emojis, QR codes, or clutter.\n"
        "- Use **legible real fonts only** (no symbols, no warped text).\n"
        "- Text should be clear, spelled correctly, and **minimal** — remove all gibberish or unreadable words.\n\n"

        " Avoid:\n"
        "- No Instagram/phone mockups, no likes, comments, usernames\n"
        "- No emojis, buttons, fake app UI, or icons unless essential\n"
        "- No clipped words, spelling errors, or random letters\n"
        "- Do not simulate camera photos or overlay frames — this is **not** a product photography layout.\n\n"

        f"Only include this heading clearly:\n"
        f"\"{slogan}\"\n"
        "- Use only this slogan. No additional headings, taglines, or fake words."
    )

def generate_prompt_from_editor(persona_summary, persona, api_key,
                                objective="", industry="", funnel_stage="", past_engagement="",
                                platform="Instagram", post_type="Text", tone="Friendly", num_variants=1):
    """
    Generate a marketing prompt from manually entered form values and persona.
    This version is designed to work without join_year, location, or loyalty fields.
    """
    image_urls = None

    prompt = f"""You are a creative marketing copywriter.

Your job is to generate a short, engaging promotion message tailored to the following campaign context and audience persona.

 Marketing Objective: {objective or "General Engagement"}
 Industry: {industry or "General"}
 Platform: {platform}
 Post Type: {post_type}
 Tone Preference: {tone}
 Funnel Stage: {funnel_stage or "Unspecified"}
 Past Engagement: {past_engagement or "Unknown"}

 Target Audience Summary:
{persona_summary}

 Style Guide:
Take inspiration from the cluster persona’s writing style and interests where suitable — do not repeat them verbatim.
Use a natural, friendly tone suitable for the platform and post type.
Include emojis where appropriate. Avoid any reference to gender or specific locations unless naturally fitting.

Now write the promotional message."""

    results = []
    if api_key != "SKIP_TEXT" and post_type in ["Text", "Both"]:
        for _ in range(num_variants):
            print(f"🧠 Generating text variant {_+1}")
            results.append(get_openai_response(prompt, api_key))  
            
    if post_type in ["Image", "Both"]:
        try:
            editor_data = {
                "objective": objective,
                "industry": industry,
                "tone": tone,
                "platform": platform
            }
            slogan = generate_slogan(editor_data, api_key)
            image_prompt = build_image_prompt(editor_data, slogan)
            print("🖼️ Final Image Prompt:\n", image_prompt)
            image_urls = generate_image_content(image_prompt, api_key, platform, num_variants=num_variants)
        except Exception as e:
            print(f"❌ Image generation failed: {e}")
            image_urls = None


    used_fields = {
        "from_user_input": ["objective", "industry", "platform", "post_type", "tone", "funnel_stage", "past_engagement"],
        "from_cluster": ["Writing_Style", "Interests"],
        "optional_inputs": []
    }
    print("📤 Final text variants returned:")
    for i, variant in enumerate(results):
        print(f"🧾 Variant {i + 1}:\n{variant}\n")
    if image_urls:
        print("🖼️ Image URLs:")
        for img in image_urls:
            print(img)
    else:
        print("🖼️ No image URLs generated.")
        
    return prompt, results, used_fields, image_urls
