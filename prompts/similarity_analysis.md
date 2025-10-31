Analyze these two responses and provide a semantic similarity analysis in JSON format.

Response 1: {response1}
Response 2: {response2}

Return ONLY valid JSON (no markdown formatting, no code blocks) with this exact structure:
{{
  "similarity_score": <number 0-100>,
  "shared_concepts": ["concept1", "concept2", ...],
  "unique_to_response1": ["unique point 1", "unique point 2", ...],
  "unique_to_response2": ["unique point 1", "unique point 2", ...]
}}

Guidelines:
- similarity_score: 0-100, where 100 means identical content
- shared_concepts: Key ideas, topics, or information present in both responses
- unique_to_response1: Important points only in Response 1
- unique_to_response2: Important points only in Response 2
- Keep each item concise (3-8 words max)
- Focus on substantive differences, not stylistic ones
