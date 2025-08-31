# 🧠 Mnemosyne - AstrBot long-term memory center

> *"Memory is the process of retaining information over time."*
> *"Memory is the means by which we draw on our past experiences in order to use this information in the present."*
> — (Paraphrased concepts based on memory research, attributing specific short quotes can be tricky)
>
> **let AI truly remember every conversation with you，build a lasting personalized experience。**

---

## 💬 support and discussion

encounter problems or want to share usage experiences？join our discussion group：
[![joinQQgroup](https://img.shields.io/badge/QQgroup-953245617-blue?style=flat-square&logo=tencent-qq)](https://qm.qq.com/cgi-bin/qm/qr?k=WdyqoP-AOEXqGAN08lOFfVSguF2EmBeO&jump_from=webapi&authKey=tPyfv90TVYSGVhbAhsAZCcSBotJuTTLf03wnn7/lQZPUkWfoQ/J8e9nkAipkOzwh)

here，you can directly communicate with developers and other users，get more timely help。

---

## 🎉 update overview

we continuously improve，here are the recent update highlights and important milestones of this plugin：

### 🚀 v0.5.0 (latest version)

*   🔗 **ecosystem compatibility:** added support for [astrbot_plugin_embedding_adapter](https://github.com/TheAnyan/astrbot_plugin_embedding_adapter) plugin compatibility support，can now interact with this plugin，get better embedding results。special thanks [@TheAnyan](https://github.com/TheAnyan) for the contribution！
*   ⚡️ **optimization and fixes:** performed multiple internal optimizations，and fixed several known issues，improved overall stability and user experience。
*   ⚖️ **protocol update:** the open-source protocol of the plugin has been changed，please refer to the `LICENSE` file in the project root directory for details。

### 🚀 v0.4.1

*   🐛 **Bug fix:** fixed in certain specific environments（such as pymilvus 2.5.4）under，processing Milvus search results may cause `TypeError: 'SequenceIterator' object is not iterable` issues。special thanks [@HikariFroya](https://github.com/HikariFroya) discovered and contributed a solution！
*   ✨ **command optimization:** simplified `/memory list_records` the use of commands，making it more focused on querying the latest memory records。
    *   command format changed to：`/memory list_records [collection_name] [limit]`，**removed `offset` parameters**。
    *   now，you only need to specify the number of records to view (`limit`)，the system will automatically retrieve all qualifying records（within the safety limit），and select the latest few to display in reverse chronological order，no need to manually calculate the offset，greatly improved convenience。
*   ✨ **model support:** the embedded model now adds support for Google Gemini support for the embedded model。thanks to [@Yxiguan](https://github.com/Yxiguan) for providing the key code！

### 🚀 v0.4.0

*   ✨ **core new features: time-based automatic summarization**:
    *   integrated timer within the plugin，when the user andBOTmessages between have not been summarized for a long time（even without new interactions），the system will automatically trigger a summary of previous historical messages，effectively reduce the frequency and omissions of manual summarization。
*   ⚙️ **new configuration items**: introduced for customizing timer interval time (`auto_summary_interval`) and summary threshold time (`auto_summary_threshold`) configuration items，users can flexibly adjust automatic summarization behavior according to needs。
*   🛠️ **architecture optimization**: refactored the context manager，optimized the storage and retrieval logic of session history，significantly improved efficiency and stability。
*   🏗️ **background tasks**: improved the start and stop logic of background automatic summary check tasks in the main program，ensure stable and reliable operation of this feature。

<details>
<summary><strong>📜 historical version review (v0.3.14 and earlier)</strong></summary>

### 🚀 v0.3.14

*   🐛 **key fixes:** resolved v0.3.13 a major issue causing data insertion failure in the version。**strongly recommend all users update to this version to ensure normal plugin operation！**

### 🚀 v0.3.13

*   ✨ **new features:** added `Milvus Lite` support！can now run a lightweight vector database locally，without complex deployment of a complete Milvus service，greatly simplified the entry threshold and local development experience。（special thanks to the community group member who suggested this！）
*   ⚠️ **important notice:** `Milvus Lite` currently only supports `Ubuntu >= 20.04` and `MacOS >= 11.0` operating system environments。

### 📜 v0.3.12 and earlier versions (main optimizations and fixes)

*   ✅ **core fixes:** included multiple key Bug fix、emergency issue handling and command logic corrections，improved the stability and robustness of the plugin。
*   🔧 **performance and logic optimization:** for session history check、asynchronousIOoptimized core modules such as processing，effectively improved operational efficiency and response speed。
*   ⚙️ **configuration and feature enhancement:** updated the configuration architecture to support more customization options，and restored or optimized some feature settings from earlier versions，to meet the needs of more usage scenarios。

*this range includes multiple iterations of update content，the above is a summary of the main categories。for more detailed historical update logs，please refer to the project's Release Notes or Git Commit history。*
</details>

---

## 🚀 quick start

want to immediately experience Mnemosyne the powerful memory？please refer to our quick start guide：

➡️ **[how to correctly and quickly use this plugin (Wiki)](https://github.com/lxfight/astrbot_plugin_mnemosyne/wiki/%E5%A6%82%E4%BD%95%E6%AD%A3%E7%A1%AE%E4%B8%94%E5%BF%AB%E9%80%9F%E7%9A%84%E9%A3%9F%E7%94%A8%E6%9C%AC%E6%8F%92%E4%BB%B6)** ⬅️

---

## ⚠️ important notice：test version risk notice

❗️ **please note：this plugin is still in active development and testing phase。**

### 1. function and data risks
*   as the plugin is still rapidly iterating，the addition of new features、code refactoring or with AstrBot compatibility adjustments with the main program，**may cause system instability or data processing anomalies in some cases**。
*   current version**does not yet include**a complete automated data migration solution。this means that during major version updates，**there is a risk of losing historical memory data**。

### 2. usage suggestions
*   ✅ **strongly recommend：** before updating the plugin version，be sure to**backup important data**，including but not limited to：
    *   plugin configuration files
    *   Milvus data（if independently deployed Milvus，please refer to its backup documentation；if it is Milvus Lite，please backup `AstrBot/data/mnemosyne_data` directory）
*   🧪 **recommended actions：** if conditions permit，suggest testing in a non-production environment first（such as a test AstrBot instance）to test the new version，confirm it is correct before updating to your main environment。

> 🛡️ **data security adage:**
> *"value your data security as you would protect important relationships——after all，no one wants their digital companion to suddenly'lose memory'。"*

---

## 📦 Milvus database installation (optional)

if you do not use v0.3.13+ the newly added version `Milvus Lite` mode，or need a more powerful standalone vector database，you can choose to install Milvus：

*   **🐧 Linux (Docker):** [Milvus standalone version Docker installation guide](https://milvus.io/docs/zh/install_standalone-docker.md)
*   **💻 Windows (Docker):** [Milvus standalone version Windows Docker installation guide](https://milvus.io/docs/zh/install_standalone-windows.md)

> **tip:** for most personal users and quick experience scenarios，`Milvus Lite` (v0.3.13+) is a more convenient choice，no additional installation required。

---

## 🧩 plugin ecosystem recommendation：optimize DeepSeek API experience

**1. this plugin (Mnemosyne v0.3+ series) 🚀**

*   Mnemosyne plugin since `v0.3` series，integrated by developers **[Rail1bc](https://github.com/Rail1bc)** contributed key optimization code。
*   **core advantages**: this optimization is specifically for DeepSeek official API caching mechanism。by intelligently adjusting the content sent to API the historical conversation content，can**significantly increase cache hit rate**。this means you can more frequently reuse previous computation results，effectively**reduce Token consumption** 💰，save you API call costs。

**2. compost bin (Composting Bucket) plugin ♻️**

*   in addition to Mnemosyne for the contribution，developer **Rail1bc** also independently developed a plugin named **“compost bin” (Composting Bucket)** of AstrBot plugin。
*   **main function**: this plugin focuses on enhancing DeepSeek API cache utilization efficiency。even if you do not use Mnemosyne the memory function，can also use“compost bin”as an independent enhancement tool，further optimize cache performance，reduce unnecessary Token overhead。（“compost bin”minimal impact on user experience，mainly optimized in the background）
*   **project address**: interested users can visit for more details：
    🔗 **[astrbot_plugin_composting_bucket on GitHub](https://github.com/Rail1bc/astrbot_plugin_composting_bucket)**

> ✨ if you are DeepSeek API user，strongly recommend following and trying the tools provided by **Rail1bc** these excellent tools，make your AI experience more economical、more efficient！

---

## 🙏 acknowledgments

*   thanks to **AstrBot core development team** for providing a powerful platform and technical support。
*   thanks to **[Rail1bc](https://github.com/Rail1bc)** to DeepSeek API for the key code contributions to optimization。
*   thanks to everyone in QQ group and GitHub Issues who provided valuable opinions and feedback。

**if this project has brought you help or joy，please do not hesitate to light up Star ⭐ ！your support is my greatest motivation for continuous development and improvement！**

discover Bug？have a good idea？please feel free to [GitHub Issues](https://github.com/lxfight/astrbot_plugin_mnemosyne/issues) tell us。we take every feedback seriously。

---

## 🌟 contributors

thanks to all who contributed to Mnemosyne the project！

[![GitHub Contributors](https://img.shields.io/github/contributors/lxfight/astrbot_plugin_mnemosyne?style=flat-square)](https://github.com/lxfight/astrbot_plugin_mnemosyne/graphs/contributors)

<a href="https://github.com/lxfight/astrbot_plugin_mnemosyne/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=lxfight/astrbot_plugin_mnemosyne" alt="Contributor List" />
</a>

---

## ✨ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=lxfight/astrbot_plugin_mnemosyne)](https://github.com/lxfight/astrbot_plugin_mnemosyne)

_each one Star is a beacon for our progress！thank you for your attention！_