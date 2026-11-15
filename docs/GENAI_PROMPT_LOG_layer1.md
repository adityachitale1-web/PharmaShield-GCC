# GenAI Prompt Log — Layer 1 (Data Ingestion & Collection)

Plan §6.1 disclosure protocol. Tool: Claude (Anthropic), 15 November 2026. Everything under `layer1_ingestion/`,
`streamlit_app.py`, `notebooks/01_layer1_data_ingestion.ipynb` and the synthetic contracts in
`layer1_ingestion/data/manual/contracts/` is GenAI-generated and should carry the red dashed outline on slides.
Human-supplied input: the Master Implementation Plan PDF. Prompts are verbatim; trim before submission if the team prefers.

| # | Prompt (verbatim) | What was generated |
|---|---|---|
| 1 | "So this is the project we are working on, in the attached. Read it and understand it thoroughly. Since I am part of a team, I need to complete the first layer of the implementation plan which is the "Data Ingestion and Collection". I want you to choose the data points, refer to any website if you have to, and write a python code that actually takes care of the data ingestion and collection. The document contains the technical aspects of the multi-source data ingestion so please focus on that and write a python code that takes care of data issues such as duplicate data, outliers, and any other issues you can think of. Ensure that this is done foolproof. The instructions are mentioned very clearly and in detail in the attached document. Also, whatever deliverables you create, I need to share it onto the group. So give me the appropriate deliverables for this (python code, data sources, etc)" | Ingestion package (SCMS, openFDA, inspections, World Bank, documents, scenario), reference tables, tests, README, DATA_SOURCES |
| 2 | "Here is the github repo - https://github.com/adityachitale1-web/PharmaShield-GCC I want you to create the github files and push it into my repo." | Repository layout, root README, first commit |
| 3 | "Which app should I run?" | Run instructions |
| 4 | "I want you to fix the issue, then push the required files onto the github repo so that I can run it on streamlit." | `streamlit_app.py`, root `requirements.txt`, Streamlit config |
| 5 | "What are the manual inputs I should give the dashboard?" | Explanation of the three upload boxes |
| 6 | "Give me the manual inputs here. What files should I upload?" | `urls.csv` (51 real FDA warning-letter URLs found by web search), 11 synthetic supplier contracts, answer key |
| 7 | "Segregate the uploads into these sections." | Upload pack with one folder per box |
| 8 | "I want the final files to be given in the upload pack. I don't want to follow anything, it's just a simple process to upload the files. Please give me the final files." | Automatic download of the FDA inspections dataset |
| 9 | "Did you check the data schema? I want you to go through the implementation plan document again and ensure that there are no mistakes. […] Ensure everything in the document is followed thoroughly and then push the files to the github repo. Also give me the files I will need to upload." | Locked-schema `suppliers_matrix.csv`, rapidfuzz entity linking, probation rule, step-by-step shape / class-balance output, optional GHSC-PSM and internal loaders, PLAN_COMPLIANCE |
| 10 | "Shouldn't the data schema reflect onto the dashboard?" followed by "Do it right. Read through the attached document and understand it THOROUGHLY. ENSURE ALL STEPS ARE FOLLOWED AND NOTHING IS MISSING." | Schema tab with validation, data-feed inventory, 360° manufacturer profile, corridor features, C2 inference features, plant-level inspections, top-20 manual check, tier reference data, notebook, this log |

## What is and is not AI-generated data

* Real data: SCMS file, everything returned by openFDA, the FDA Data Dashboard and the World Bank, FDA warning letters.
* AI-generated: the 11 supplier contracts (each file states it is synthetic) and their answer key.
* Scenario assumptions written by AI and labelled SCENARIO: corridor shares, freight per vial, vial weight, INCO term,
  default capacity, plant-country hints. The team should review them before presenting.
