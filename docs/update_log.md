### v0.4.0
- **New feature**:Built-in timer in the plugin，when the user andBOTMessages have not been summarized for a long time（The user may not haveBOTInteraction），Automatically summarize previous historical messages。

### v0.3.13
- **addedmilvus litesupport**：Addedmilvus liteSupport for，Can use an extremely lightweight database locally，Without deploymentmilvus。（Thanks to the group members for the suggestions made a few days ago，But I can't find who mentioned it to me now，Can come to claim it）

---

### v0.3.4 -> v0.3.12
- **SpecifyLLMService provider configurationbugfix**：Fixed specifiedLLMInitialization error after service providerbug
- **Configuration increase**：Updated configuration architecture，Support specifiedLLMService provider for memory summarization
- **Logic optimization**：Each session will check if there are long-term memory fragments that need to be deleted in previous historical messages。
- **Function optimization**：Use asynchronous method to handle synchronizationIOoperation，Avoid blocking the main thread。
- **Function error fix**：Adjusted the case sensitivity issue of regular expressions
- **Emergency fix**：Added support for astrbot_max_context_length Size judgment，Avoid errors caused by negative numbers
- **Restore configuration item function**：Restored configuration item`Number of long-term memories retained in historical context (contexts_memory_len)`Function。
- **fix`num_pairs`Ineffective error**：Fixed about`num_pairs`Configuration item not effective，Causing memory summarization，Carry a large amount of historical records。
- **fixBUG**：About`v0.3.3`Since the version，callLLMWhen summarizing memory，Old memories will exist in the context，This version has been deleted，To solve some problems。
- **Fix commandBUG**：About`/memory list_records` Command error fix

---

### 🌱 v0.3.0
**Release date**：2025-04-03

*   **Core refactoring and optimization：** This update to Milvus Rewritten the control logic of the database，And simultaneously optimized the storage mechanism of long-term memory。
    *   **important notice：** Due to underlying refactoring，The stability of the new implementation has not been fully verified，It is recommended to use cautiously after assessing the risks。
*   **Connectivity extension：** Extended Milvus Database connection options，Added support for connecting through proxy addresses in the design。
    *   **Note：** This proxy connection feature is currently only at the design stage，Not yet tested or verified in practice。

---

### 🚀 v0.2.0
**Release date**：2025-02-23
- **Complete refactoring**:Refactor project code，Improve code scalability
- **Resource management**:toMilvusReasonable management of database connections，But the current solution is not optimal

1. **Based on personalityIDAnd sessionIDMemory distinction**
    - Now supports based on **PersonaID** and **SessionID** Distinguish memory。
    - **PersonaID** Distinction is optional（Can be enabled or disabled through configuration）。
    - **SessionID** Isolation is absolute，Ensure that memories between different sessions are completely independent。

2. **Long-term memory switching**
    - Currently does not support dynamic switching of long-term memory in a single session（We will evaluate the need and implementation plan for this feature in future versions）。

3. **Dialogue round threshold update mechanism**
    - We improved the memory logic，Now adopts **Dialogue round threshold** To trigger memory updates：
    - When the dialogue round reaches the specified threshold，The system will immediately summarize the historical dialogue content and store it as long-term memory。
    - This mechanism can effectively reduce redundant memory，While improving the accuracy and efficiency of summarization。

4. **Compatibility issues**
    - The new version and the old versionMilvusDatabase is not compatible，Need to modify after updating the version`collection_name`parameters，Use new collections

#### ⚠️ Upgrade notice
1. **Not backward compatible**：Due to architecture refactoring，MilvusThe format in the database will change，It is recommended to use in the configuration`collection_name`Switch to a new database，Unable to achieve long-term memory migration temporarily
2. **Command changes**：Due to code refactoring，Commands have also changed，Please use specifically/memory Query usage

### 🌱 v0.1.0
**Release date**：2025-02-19
- Implement basic memory storage/Retrieval function  
- supportMilvusVector database basic operations
- Build the core algorithm framework for dialogue summarization
