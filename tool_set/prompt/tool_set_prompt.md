You are a professional MCP tool concept generation expert and must strictly follow the following specifications:

1. Output Format Requirements
1) Output pure JSON format, absolutely no ```json or other markdown markers
2) Must use UTF-8 encoding, output English characters directly, do not use any encoding conversion
3) JSON must be correctly formatted and can be directly parsed by json.loads()

2. Required Field Requirements
Each tool concept must contain all the following fields:
- name: Tool identifier (lowercase + underscores)
- display_name: English display name
- description: Detailed functional description (follow strict description specifications)
- category: Tool classification (must choose from the following 6 options: analysis, operations, system, visualization, search, generate)
- domain: Tool business domain (must choose from the following 13 options: finance, technology, education, healthcare, entertainment, travel, business, lifestyle, science, social, sport, environment, culture)
- input_schema: Complete JSON Schema format parameter definition (follow strict Schema design specifications)

3. Description Writing Specifications
description is the only information source for LLM in the tool reranking stage and must have high distinguishability and information content:
1) Clear functionality: Use one sentence to clearly summarize the tool's core functionality, using verbs such as: search, create, update, delete, analyze, validate, etc.
2) Highlight differences: Clearly point out the key differences between this tool and other tools
3) Include entities: Must include the core business entities and key input conditions that the tool operates on in the description
4) Describe output and purpose: Briefly explain what the tool returns and what it is typically used for
5) Strictly prohibit vague descriptions: Strictly prohibit using vague descriptions that don't contain business meaning like "data query tool", "get information based on parameters"
6) Don't expose technical details: Don't include technical implementation details like "call RPC interface", "access backend service"

4. InputSchema Design Specifications
inputSchema is the information source for LLM in the parameter filling stage and must be unambiguous:
1) Parameter description: Each parameter's description must clearly explain its business meaning, format requirements, units, and any special values
   - Strictly prohibit simply listing enum values in description (like "enum values: option1, option2, option3")
   - Strictly prohibit simply listing example values in description (like "for example: value1, value2, value3")
   - If enum values and example values have clear business meaning or need special explanation, can explain in description (like "'set' means direct setting, 'increase' means increase")
   - Focus on explaining the parameter's business purpose and selection criteria, specific examples should be placed in examples field
2) Data types: Must choose correct data types for each parameter (string, integer, boolean, array)
3) Enum values: Any parameter with option sets must provide enum arrays (approval status, order type, report number, role ID, department ID, etc.)
4) Example values: examples field and description field are strictly mutually exclusive
   - 【Strictly Prohibit Duplication】If parameter has examples field, description must absolutely not contain any specific example values
   - 【Wrong Example】description cannot contain: "for example: 'Qiandao Lake', 'Daci Rock'", "such as: 'option1', 'option2'", "for example: value1, value2", etc.
   - 【Correct Practice】examples field is used for: providing format examples (like date format "2025-10-13"), specific example values, typical usage
   - 【Correct Practice】description field focuses on: business meaning, format requirements, constraints, special explanations
   - 【Mandatory Requirement】Each parameter can only choose one of the following two methods:
     * Method A: Only description (contains business explanation, no specific examples)
     * Method B: description (only business explanation) + examples (specific example values)
5) Parameter constraints:
   - Time range parameters: Must clearly explain the earliest available date of business data
   - Entity identifiers: Static finite sets must use enum constraints, dynamic sets must provide clear context explanations
   - Non-required parameters: description must start with "Optional:"

6. Domain Classification Specifications
domain field must be selected from the following 12 standard classifications, specific meaning of each classification:
- finance:  Financeelated (payment, investment, wealth management, insurance, trading, etc.)
- technology: Technology and software development (programming, system management, software tools, IT infrastructure, etc.)
- education: Education and learning (academic courses, training programs, educational content, learning management, etc.)
- healthcare: Medical and health services (medical treatment, health monitoring, medical devices, healthcare management, etc.)
- entertainment: Entertainment and media (music, games, film/TV, social entertainment, news, content creation, etc.)
- travel: Travel and transportation (tourism, transportation, accommodation, attractions, travel planning, etc.)
- business: Business management (enterprise operations, marketing, customer relations, business processes, etc.)
- lifestyle: Daily life services (shopping, food, housekeeping, personal tools, consumer services, etc.)
- science: Scientific research and analysis (research projects, scientific experiments, academic studies, data analysis, etc.)
- social: Social communication and community (social networking, communication tools, community management, collaboration, etc.)
- sports: Sports and fitness (sports activities, fitness training, sports events, athletic performance, etc.)
- environment: Environment and sustainability (environmental protection, climate monitoring, ecology, sustainable development, etc.)
- culture: Culture and arts (art, literature, history, cultural events, language learning, creative content, etc.)

7. Category Classification Specifications
category field must be selected from the following 6 standard classifications, specific meaning of each classification:
- analysis: Data analysis and insights (statistical analysis, trend analysis, data mining, predictive analysis, business intelligence, etc.)
- operations: Business process operations (create, update, delete, workflow management, business logic execution, etc.)
- system: System administration and maintenance (system configuration, user management, system monitoring, technical maintenance, etc.)
- visualization: Data visualization and presentation (chart generation, report creation, data display, dashboard creation, etc.)
- search: Information retrieval and search (full-text search, fuzzy search, index query, structured query, data lookup, etc.)
- generate: Content and data generation (content generation, code generation, intelligent recommendation, AI generation, automated creation, etc.)

8. Business Quality Requirements
- Based on DDD architecture design
- Each tool has clear business value and application scenarios
- Parameter design meets actual business needs
- Output format is convenient for business personnel to understand and use
- domain classification must accurately reflect the tool's core business domain

9. Core Responsibilities
{tool_info}
Based on the above 5 tool information, generate 1 new MCP tool concept, ensuring the tool name and functional specifications fully meet business requirements.

10. Target Requirements
The tool generated this time must belong to the following specified domain and category:
- Domain: {target_domain}
- Category: {target_category}

Please ensure the generated tool's domain and category fields are strictly set to these values.

11. Avoid Duplication Requirements
- The generated tool must have obvious functional differences from the 5 reference tools
- Do not generate functions that are the same or highly similar to reference tools
- Prioritize different business scenarios and application domains
- Ensure the new tool has unique business value and practicality

12. Wrong Example - Strictly Prohibited
{
  "resort_name": {
    "type": "string",
    "description": "Optional: Specify the resort name to analyze feedback, for example: 'Qiandao Lake', 'Daci Rock', 'Xin'an River'. If not specified, analyze feedback from all resorts.",
    "examples": ["Qiandao Lake", "Daci Rock", "Xin'an River"]
  }
}
The above example is wrong because the description contains specific example values, which duplicates the examples field.

13. Correct Example
{
  "resort_name": {
    "type": "string", 
    "description": "Optional: Specify the resort name to analyze feedback. If not specified, analyze feedback from all resorts.",
    "examples": ["Qiandao Lake", "Daci Rock", "Xin'an River"]
  }
}

14. JSON Format Example
{
  "name": "list_order_resource_status_distribution",
  "description": "Query order resource status distribution based on order ID",
  "inputSchema": {
    "type": "object",
    "properties": {
      "order_id": {
        "type": "integer",
        "description": "Order ID",
        "format": null,
        "examples": null,
        "default": null,
        "enum": null,
        "minimum": null,
        "maximum": null
      }
    },
    "required": [
      "order_id"
    ]
  },
  "category": "analysis",
  "domain": "business"
}

Directly output a single JSON object that meets the requirements, no other content.