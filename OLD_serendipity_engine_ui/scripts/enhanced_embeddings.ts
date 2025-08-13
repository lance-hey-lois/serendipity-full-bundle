/**
 * Enhanced Embedding Strategy for Better Domain Matching
 * ========================================================
 * This shows how to improve embeddings for better search accuracy
 */

// CURRENT APPROACH (from your codebase):
function currentEmbedding(profile: any): string {
  return `
    Name: ${profile.name || ''}
    Title: ${profile.title || ''}
    Company: ${profile.company || ''}
    Location: ${profile.locatedIn || ''}
    Short Pitch: ${profile.shortPitch || ''}
    Blurb: ${profile.blurb || ''}
    Goals: ${profile.midTermPriorities?.goals?.join(', ') || ''}
    Looking For: ${profile.midTermPriorities?.lookingFor?.join(', ') || ''}
    Network Strengths: ${profile.areasOfNetworkStrength?.join(', ') || ''}
    Can Help With: ${profile.canHelpWith?.join(', ') || ''}
    Interests: ${profile.passionsAndInterests?.join(', ') || ''}
    Personality Traits: ${profile.personalityTraits?.join(', ') || ''}
    Relevant Details: ${profile.relevantDetails || ''}
  `.trim();
}

// ENHANCED APPROACH - Better domain specificity:
function enhancedEmbedding(profile: any): string {
  // 1. Detect industry/domain from title, company, skills
  const industry = detectIndustry(profile);
  
  // 2. Create domain-specific context
  const domainContext = industry ? `Industry: ${industry}. ` : '';
  
  // 3. Emphasize role-specific keywords
  const roleEmphasis = extractRoleKeywords(profile.title);
  
  // 4. Build enhanced embedding with better signal
  return `
    ${domainContext}
    ${profile.name} is a ${profile.title} at ${profile.company}.
    Professional focus: ${profile.areasOfNetworkStrength?.join(', ')}.
    ${roleEmphasis ? `Role keywords: ${roleEmphasis}.` : ''}
    Bio: ${profile.blurb}
    Expertise: ${profile.canHelpWith?.join(', ')}.
    Goals: ${profile.midTermPriorities?.goals?.join(', ')}.
    Seeking: ${profile.midTermPriorities?.lookingFor?.join(', ')}.
    Interests: ${profile.passionsAndInterests?.join(', ')}.
    ${profile.relevantDetails || ''}
  `.trim();
}

// Industry detection based on keywords
function detectIndustry(profile: any): string | null {
  const musicKeywords = ['music', 'song', 'composer', 'musician', 'artist', 'band', 'record', 'album', 'studio', 'broadway', 'theater', 'juke'];
  const techKeywords = ['software', 'developer', 'engineer', 'tech', 'AI', 'ML', 'data', 'platform', 'SaaS'];
  const financeKeywords = ['investment', 'capital', 'fund', 'finance', 'banking', 'venture', 'equity'];
  const healthKeywords = ['health', 'medical', 'cancer', 'care', 'hospital', 'clinical', 'therapy'];
  
  const allText = `${profile.title} ${profile.company} ${profile.areasOfNetworkStrength?.join(' ')} ${profile.blurb}`.toLowerCase();
  
  if (musicKeywords.some(kw => allText.includes(kw))) return 'Music & Entertainment';
  if (techKeywords.some(kw => allText.includes(kw))) return 'Technology';
  if (financeKeywords.some(kw => allText.includes(kw))) return 'Finance & Investment';
  if (healthKeywords.some(kw => allText.includes(kw))) return 'Healthcare';
  
  return null;
}

// Extract role-specific keywords for better matching
function extractRoleKeywords(title: string): string {
  if (!title) return '';
  
  const roleMap: Record<string, string> = {
    'songwriter': 'music composition, lyrics, creative writing, music industry',
    'composer': 'music composition, orchestration, arrangement, music creation',
    'musician': 'music performance, instruments, recording, live music',
    'ceo': 'leadership, strategy, management, business development',
    'engineer': 'technical, development, systems, architecture',
    'investor': 'capital, funding, portfolio, returns',
    'designer': 'visual, UX, interface, creative design',
  };
  
  const lowerTitle = title.toLowerCase();
  for (const [key, keywords] of Object.entries(roleMap)) {
    if (lowerTitle.includes(key)) {
      return keywords;
    }
  }
  
  return '';
}

// ALTERNATIVE: Structured embedding with clear sections
function structuredEmbedding(profile: any): string {
  return `
    [PROFESSIONAL]
    ${profile.title} at ${profile.company}
    Industry: ${detectIndustry(profile) || 'General'}
    
    [EXPERTISE]
    ${profile.areasOfNetworkStrength?.join(', ')}
    ${profile.canHelpWith?.join(', ')}
    
    [BACKGROUND]
    ${profile.blurb}
    
    [GOALS]
    ${profile.midTermPriorities?.goals?.join(', ')}
    ${profile.midTermPriorities?.lookingFor?.join(', ')}
    
    [PERSONAL]
    ${profile.passionsAndInterests?.join(', ')}
    ${profile.personalityTraits?.join(', ')}
  `.trim();
}

export { currentEmbedding, enhancedEmbedding, structuredEmbedding, detectIndustry };

/**
 * RECOMMENDATIONS:
 * 
 * 1. Add explicit industry/domain detection before embedding
 * 2. Use natural language sentences instead of "Property: value" format
 * 3. Emphasize role-specific keywords that distinguish professions
 * 4. Consider creating separate embedding fields:
 *    - embedding_general: Full profile embedding
 *    - embedding_role: Just title + skills + industry
 *    - embedding_bio: Just the bio text
 * 
 * 5. For search, you could:
 *    - First filter by industry/domain
 *    - Then do semantic search within that domain
 *    - Or weight different embedding fields differently
 * 
 * The key issue is that generic embeddings treat all text equally,
 * so "conflict resolution" and "songwriting" might have similar
 * abstract patterns even though they're completely different domains.
 */