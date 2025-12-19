# 提示词工程 (Prompt Engineering)

**作者：** Lee Boonstra
**日期：** 2024年9月

---

## 致谢 (Acknowledgements)

**审稿人与贡献者**
Michael Sherman, Yuan Cao, Erick Armbrust, Anant Nawalgaria, Antonio Gulli, Simone Cammel

**策划与编辑**
Antonio Gulli, Anant Nawalgaria, Grace Mollison

**技术文档工程师**
Joey Haymaker

**设计师**
Michael Lanning

---

## 目录

*   引言
*   提示词工程
*   LLM 输出配置
    *   输出长度
    *   采样控制（Temperature, Top-K 和 Top-P）
    *   综合运用
*   提示技术
    *   通用提示 / 零样本 (Zero shot)
    *   单样本 (One-shot) & 少样本 (Few-shot)
    *   系统、上下文和角色提示
    *   后退一步提示 (Step-back prompting)
    *   思维链 (Chain of Thought, CoT)
    *   自我一致性 (Self-consistency)
    *   思维树 (Tree of Thoughts, ToT)
    *   ReAct (推理与行动)
    *   自动提示工程 (Automatic Prompt Engineering)
*   代码提示
    *   编写代码、解释代码、翻译代码、调试与审查代码
    *   关于多模态提示？
*   最佳实践
*   总结
*   尾注

---

# 引言

> “你不需要是数据科学家或机器学习工程师——每个人都可以编写提示词。”

当思考大语言模型（LLM）的输入与输出时，文本提示词（Prompt，有时伴随图像提示词等多模态形式）正是模型用来预测特定输出的输入内容。你不需要具备数据科学家或机器学习工程师的背景——每个人都能写提示词。然而，撰写**最有效**的提示词可能颇具复杂性。提示词的诸多方面都会影响其功效：你使用的模型、模型的训练数据、模型配置、你的遣词造句、风格语气、结构以及上下文背景，所有这些都至关重要。

因此，提示词工程是一个迭代的过程。不恰当的提示词会导致模棱两可、不准确的回复，并阻碍模型提供有意义的输出。

当你在与 Gemini 聊天机器人对话时，你本质上是在编写提示词。但本白皮书主要聚焦于在 Vertex AI 中或通过 API 使用 Gemini 模型时的提示词编写，因为通过直接提示模型，你可以访问诸如温度（Temperature）等配置参数。

本白皮书将详细探讨提示词工程。我们将深入研究各种提示技术以助你入门，并分享技巧和最佳实践，助你成为提示词专家。我们还将讨论在打磨提示词时可能面临的一些挑战。

# 提示词工程 (Prompt Engineering)

请记住 LLM 的工作原理：它是一个**预测引擎**。模型接收序列文本作为输入，然后根据其训练数据预测下一个 Token（词元）应该是什么。LLM 被设计为一遍又一遍地执行此操作，将先前预测的 Token 添加到序列文本的末尾，以预测再下一个 Token。下一个 Token 的预测是基于先前 Token 中的内容与 LLM 在训练期间所见内容之间的关系。

当你编写提示词时，你是在试图设定 LLM 以预测正确的 Token 序列。**提示词工程是设计高质量提示词以引导 LLM 产生准确输出的过程。** 这个过程涉及反复调试以找到最佳提示词、优化提示词长度，并根据任务评估提示词的写作风格和结构。在自然语言处理和 LLM 的语境下，提示词是提供给模型以生成响应或预测的输入。

这些提示词可用于实现各种理解和生成任务，如文本摘要、信息提取、问答、文本分类、语言或代码翻译、代码生成以及代码文档撰写或推理。

请随意参考 Google 的提示指南，其中包含简单有效的提示示例。

在进行提示词工程时，首先要选择一个模型。无论你是使用 Vertex AI 中的 Gemini 语言模型，还是 GPT、Claude，亦或是 Gemma 或 LLaMA 等开源模型，提示词可能都需要针对特定模型进行优化。

除了提示词之外，你还需要调试 LLM 的各种配置。

# LLM 输出配置

一旦选定模型，你就需要确定模型配置。大多数 LLM 都提供了各种控制输出的配置选项。有效的提示词工程需要针对你的任务优化这些设置。

## 输出长度 (Output length)

一个重要的配置设置是响应中生成的 Token 数量。生成更多的 Token 需要 LLM 进行更多的计算，从而导致更高的能耗、可能更慢的响应时间以及更高的成本。

减少 LLM 的输出长度并不会导致 LLM 在其创建的输出中变得在风格或文本上更简洁，它只是强制 LLM 在达到限制后停止预测更多的 Token。如果你的需求需要简短的输出长度，你可能还需要通过工程化你的提示词来适应。

输出长度限制对于某些 LLM 提示技术尤为重要，例如 ReAct，在这种技术中，LLM 可能会在你想要的响应之后继续输出无用的 Token。

## 采样控制 (Sampling controls)

LLM 在形式上并不预测单一的 Token。相反，LLM 预测下一个 Token 可能是什么的概率，LLM 词汇表中的每个 Token 都会获得一个概率。然后对这些 Token 概率进行采样，以确定下一个生成的 Token 具体是什么。

**Temperature（温度）**、**Top-K** 和 **Top-P** 是最常见的配置设置，决定了如何处理预测的 Token 概率以选择单个输出 Token。

### 温度 (Temperature)

温度控制 Token 选择的随机程度。
*   **较低的温度**适合期望更具确定性回复的提示词。
*   **较高的温度**会导致更多样化或意想不到的结果。
*   **温度为 0**（贪婪解码）是确定性的：总是选择概率最高的 Token（不过请注意，如果两个 Token 具有相同的最高预测概率，根据打破平局的实现方式，温度为 0 时可能并不总是得到完全相同的输出）。

接近最大值的温度往往会产生更随机的输出。随着温度越来越高，所有 Token 成为下一个预测 Token 的可能性变得均等。

Gemini 的温度控制可以理解为类似于机器学习中使用的 Softmax 函数。低温度设置反映了低 Softmax 温度 (T)，强调具有高确定性的单一首选结果。较高的 Gemini 温度设置就像高 Softmax 温度，使得围绕所选设置的更广泛的结果范围变得更容易被接受。这种增加的不确定性适应了不需要严格、精确结果的场景，例如在尝试创意输出时。

### Top-K 和 Top-P

Top-K 和 Top-P（也称为核采样，Nucleus Sampling）是 LLM 中使用的两种采样设置，用于限制下一个预测的 Token 仅来自具有最高预测概率的 Token 集合。像温度一样，这些采样设置控制生成文本的随机性和多样性。

*   **Top-K 采样**：从模型预测分布中选择前 K 个最可能的 Token。Top-K 越高，模型的输出就越具创造性和变化；Top-K 越低，模型的输出就越保守和实事求是。Top-K 为 1 等同于贪婪解码。
*   **Top-P 采样**：选择累积概率不超过特定值 (P) 的前几个 Token。P 的值范围从 0（贪婪解码）到 1（LLM 词汇表中的所有 Token）。

在 Top-K 和 Top-P 之间进行选择的最佳方法是尝试这两种方法（或同时使用两者），看看哪一种能产生你想要的结果。

## 综合运用

选择 Top-K、Top-P、温度和生成的 Token 数量取决于具体的应用和期望的结果，并且这些设置都会相互影响。确保你了解所选模型如何结合不同的采样设置也很重要。

*   如果 Temperature、Top-K 和 Top-P **都可用**（如在 Vertex Studio 中）：首先筛选出同时满足 Top-K 和 Top-P 标准的 Token 候选者，然后应用 Temperature 从这些通过筛选的 Token 中进行采样。
*   如果**只有** Top-K 或 Top-P 可用：行为相同，只是仅使用该单一设置。
*   如果 Temperature **不可用**：则从满足 Top-K 和/或 Top-P 标准的 Token 中随机选择，生成下一个预测 Token。

在一种采样配置值的极端设置下，该采样设置可能会抵消其他配置设置或使其变得无关紧要：

*   **如果将 Temperature 设为 0**：Top-K 和 Top-P 就变得无关紧要——最可能的 Token 将成为下一个预测的 Token。
*   **如果将 Temperature 设得极高**（通常超过 1，甚至到 10 以上）：Temperature 变得无关紧要，任何通过 Top-K 和/或 Top-P 标准的 Token 都有可能被随机选中。
*   **如果将 Top-K 设为 1**：Temperature 和 Top-P 变得无关紧要。只有一个 Token 通过 Top-K 标准，该 Token 即为下一个预测结果。
*   **如果将 Top-K 设得极高**（如 LLM 词汇表的大小）：任何非零概率的 Token 都满足 Top-K 标准，实际上没有进行筛选。
*   **如果将 Top-P 设为 0**（或极小值）：大多数 LLM 实现将只考虑满足 Top-P 标准的最可能的 Token，使 Temperature 和 Top-K 无关紧要。
*   **如果将 Top-P 设为 1**：任何非零概率的 Token 都满足 Top-P 标准，实际上没有进行筛选。

> **建议的起始配置：**
>
> *   **一般连贯结果**：Temperature 0.2, Top-P 0.95, Top-K 30。这能提供相对连贯且适度创意的结果。
> *   **高创意结果**：Temperature 0.9, Top-P 0.99, Top-K 40。
> *   **低创意/保守结果**：Temperature 0.1, Top-P 0.9, Top-K 20。
> *   **单一正确答案（如数学题）**：Temperature 0。

**注意**：自由度越高（更高的温度、Top-K、Top-P 和输出 Token），LLM 生成的文本可能相关性越低。

---

# 提示技术

LLM 经过调整以遵循指令，并在大量数据上进行训练，以便它们能够理解提示词并生成答案。但 LLM 并不完美；你的提示文本越清晰，LLM 预测下一个可能文本的效果就越好。此外，利用 LLM 训练方式和工作原理的特定技术将帮助你从 LLM 获得相关结果。

现在我们理解了什么是提示词工程及其要素，让我们深入探讨一些最重要的提示技术示例。

## 通用提示 / 零样本 (Zero shot)

**零样本 (Zero-shot)** 提示是最简单的提示类型。它只提供任务描述和一些供 LLM 开始的文本。这个输入可以是任何东西：一个问题、故事的开头或指令。名称“零样本”代表“没有示例”。

让我们使用 Vertex AI 中的 Vertex AI Studio (Language)，它提供了一个测试提示词的游乐场。在表 1 中，你将看到一个用于分类电影评论的零样本提示示例。

> **关于表格格式的说明**：
> 下文使用的表格格式是记录提示词的绝佳方式。你的提示词在最终进入代码库之前可能会经历多次迭代，因此以一种有纪律、结构化的方式跟踪提示词工程工作非常重要。关于此表格格式、跟踪工作的重要性以及提示词开发流程的更多信息，将在本章后面的“最佳实践”部分（“记录各种提示尝试”）中详细介绍。

模型 Temperature 应设置为较低的数字，因为不需要创造力；我们使用 gemini-pro 默认的 Top-K 和 Top-P 值（这实际上禁用了这两个设置）。请注意生成的输出。词语 *disturbing*（令人不安）和 *masterpiece*（杰作）在同一句话中使用，这可能会使预测变得稍微复杂一些。

**表 1. 零样本提示示例**

| 项目            | 内容                                                         |
| :-------------- | :----------------------------------------------------------- |
| **Name**        | 1_1_movie_classification                                     |
| **Goal**        | 将电影评论分类为正面、中立或负面。                           |
| **Model**       | gemini-pro                                                   |
| **Temperature** | 0.1                                                          |
| **Token Limit** | 5                                                            |
| **Top-K**       | N/A                                                          |
| **Top-P**       | 1                                                            |
| **Prompt**      | Classify movie reviews as POSITIVE, NEUTRAL or NEGATIVE.<br>Review: "Her" is a disturbing study revealing the direction humanity is headed if AI is allowed to keep evolving, unchecked. I wish there were more movies like this masterpiece.<br>Sentiment: |
| **Output**      | POSITIVE                                                     |

当零样本不起作用时，你可以在提示词中提供演示或示例，这就引出了“单样本”和“少样本”提示。

## 单样本 (One-shot) & 少样本 (Few-shot)

在为 AI 模型创建提示词时，提供示例非常有帮助。这些示例可以帮助模型理解你的要求。当你希望引导模型生成特定的输出结构或模式时，示例尤为有用。

*   **单样本提示 (One-shot prompt)**：提供**单个**示例，故名“单样本”。其核心思想是让模型有一个可以模仿的例子，以最好地完成任务。
*   **少样本提示 (Few-shot prompt)**：提供**多个**示例。这种方法向模型展示了它需要遵循的模式。理念与单样本相似，但所需模式的多个示例增加了模型遵循该模式的机会。

少样本提示所需的示例数量取决于几个因素，包括任务的复杂性、示例的质量以及你使用的生成式 AI (gen AI) 模型的能力。**作为一般的经验法则，你应该为少样本提示使用至少 3 到 5 个示例。** 但是，对于更复杂的任务，你可能需要使用更多示例，或者由于模型的输入长度限制，你可能需要使用更少的示例。

表 2 展示了一个少样本提示示例，除了增加 Token 限制以适应更长的响应外，我们使用与之前相同的 gemini-pro 模型配置设置。

**表 2. 少样本提示示例**

| 项目            | 内容                                                         |
| :-------------- | :----------------------------------------------------------- |
| **Goal**        | 将披萨订单解析为 JSON                                        |
| **Model**       | gemini-pro                                                   |
| **Temperature** | 0.1                                                          |
| **Token Limit** | 250                                                          |
| **Top-K**       | N/A                                                          |
| **Top-P**       | 1                                                            |
| **Prompt**      | Parse a customer's pizza order into valid JSON:<br><br>EXAMPLE:<br>I want a small pizza with cheese, tomato sauce, and pepperoni.<br>JSON Response:<br>\`\`\`<br>{<br>"size": "small",<br>"type": "normal",<br>"ingredients": [["cheese", "tomato sauce", "peperoni"]]<br>}<br>\`\`\`<br><br>EXAMPLE:<br>Can I get a large pizza with tomato sauce, basil and mozzarella<br>{<br>"size": "large",<br>"type": "normal",<br>"ingredients": [["tomato sauce", "bazel", "mozzarella"]]<br>}<br><br>Now, I would like a large pizza, with the first half cheese and mozzarella. And the other tomato sauce, ham and pineapple.<br>JSON Response: |
| **Output**      | \`\`\`<br>{<br>"size": "large",<br>"type": "half-half",<br>"ingredients": [["cheese", "mozzarella"], ["tomato sauce", "ham", "pineapple"]]<br>}<br>\`\`\` |

当为提示词选择示例时，请使用与你想要执行的任务**相关**的示例。示例应多样化、高质量且书写良好。一个小错误可能会混淆模型并导致不期望的输出。

如果你试图生成对各种输入都稳健的输出，那么在示例中包含**边缘情况 (edge cases)** 很重要。边缘情况是指不寻常或意外的输入，但模型仍应能够处理。

## 系统、上下文和角色提示

系统、上下文和角色提示都是用于指导 LLM 如何生成文本的技术，但它们侧重于不同的方面：

*   **系统提示 (System prompting)**：设定语言模型的整体背景和目的。它定义了模型应该做什么的“大局”，例如翻译语言、分类评论等。
*   **上下文提示 (Contextual prompting)**：提供与当前对话或任务相关的具体细节或背景信息。它有助于模型理解所提问题的细微差别，并相应地调整回复。
*   **角色提示 (Role prompting)**：为语言模型分配一个特定的角色或身份。这有助于模型生成与分配的角色及其相关知识和行为一致的回复。

系统、上下文和角色提示之间可能存在相当大的重叠。例如，一个分配角色给系统的提示词，也可能包含上下文。

然而，每种类型的提示词都有其略微不同的主要目的：
*   **系统提示**：定义模型的基本能力和总体目标。
*   **上下文提示**：提供即时的、任务特定的信息以指导回复。它高度针对当前的任务或输入，是动态的。
*   **角色提示**：构建模型的输出风格和声音。它增加了一层特异性和个性。

区分系统、上下文和角色提示为设计具有明确意图的提示词提供了一个框架，允许灵活组合，并使分析每种提示类型如何影响语言模型的输出变得更加容易。让我们深入了解这三种不同类型的提示。

### 系统提示 (System prompting)

表 3 包含一个系统提示，我在其中指定了有关如何返回输出的附加信息。我提高了 Temperature 以获得更高的创造力水平，并指定了更高的 Token 限制。然而，由于我就如何返回输出给出了明确的指令，模型没有返回多余的文本。

**表 3. 系统提示示例**

| 项目            | 内容                                                         |
| :-------------- | :----------------------------------------------------------- |
| **Goal**        | 将电影评论分类为正面、中立或负面。                           |
| **Model**       | gemini-pro                                                   |
| **Temperature** | 1                                                            |
| **Token Limit** | 5                                                            |
| **Top-K**       | 40                                                           |
| **Top-P**       | 0.8                                                          |
| **Prompt**      | Classify movie reviews as positive, neutral or negative. Only return the label in uppercase.<br>Review: "Her" is a disturbing study revealing the direction humanity is headed if AI is allowed to keep evolving, unchecked. It's so disturbing I couldn't watch it.<br>Sentiment: |
| **Output**      | NEGATIVE                                                     |

系统提示对于生成满足特定要求的输出非常有用。“系统提示”这个名字实际上代表“向系统提供额外的任务”。例如，你可以使用系统提示生成与特定编程语言兼容的代码片段，或者你可以使用系统提示返回某种结构。请看表 4，我要求以 JSON 格式返回输出。

**表 4. 带 JSON 格式的系统提示示例**

| 项目            | 内容                                                         |
| :-------------- | :----------------------------------------------------------- |
| **Goal**        | 将电影评论分类为正面、中立或负面，返回 JSON。                |
| **Model**       | gemini-pro                                                   |
| **Temperature** | 1                                                            |
| **Token Limit** | 1024                                                         |
| **Top-K**       | 40                                                           |
| **Top-P**       | 0.8                                                          |
| **Prompt**      | Classify movie reviews as positive, neutral or negative. Return valid JSON:<br>Review: "Her" is a disturbing study revealing the direction humanity is headed if AI is allowed to keep evolving, unchecked. It's so disturbing I couldn't watch it.<br>Schema:<br>\`\`\`<br>MOVIE:<br>{<br>"sentiment": String "POSITIVE" \| "NEGATIVE" \| "NEUTRAL",<br>"name": String<br>}<br>MOVIE REVIEWS:<br>{<br>"movie_reviews": [MOVIE]<br>}<br>\`\`\`<br>JSON Response: |
| **Output**      | \`\`\`<br>{<br>"movie_reviews": [<br>{<br>"sentiment": "NEGATIVE",<br>"name": "Her"<br>}<br>]<br>}<br>\`\`\` |

从提取数据的提示中返回 JSON 对象有一些好处。在实际应用中，我不需要手动创建这种 JSON 格式，我已经可以以排序的顺序返回数据（在处理日期时间对象时非常方便），但最重要的是，通过提示生成 JSON 格式，它**强制模型创建一个结构并限制幻觉 (hallucinations)**。

系统提示对于安全性和毒性控制也非常有用。要控制输出，只需在提示词中添加额外的一行，如：“You should be respectful in your answer. (你应该在回答中保持尊重。)”

### 角色提示 (Role prompting)

角色提示是一种提示词工程技术，涉及为生成式 AI 模型分配特定角色。这有助于模型生成更相关和信息丰富的输出，因为模型可以根据分配给它的特定角色来调整其响应。

例如，你可以通过角色提示让生成式 AI 模型成为图书编辑、幼儿园老师或励志演说家。一旦模型被分配了一个角色，你就可以给它特定于该角色的提示。例如，你可以提示老师制定一个教案。请看表 5，模型扮演了导游的角色。

**表 5. 角色提示示例**

| 项目            | 内容                                                         |
| :-------------- | :----------------------------------------------------------- |
| **Goal**        | 扮演导游并提供 3 个旅行建议                                  |
| **Model**       | gemini-pro                                                   |
| **Temperature** | 1                                                            |
| **Token Limit** | 1024                                                         |
| **Top-K**       | 40                                                           |
| **Top-P**       | 0.8                                                          |
| **Prompt**      | I want you to act as a travel guide. I will write to you about my location and you will suggest 3 places to visit near me. In some cases, I will also give you the type of places I will visit.<br>My suggestion: "I am in Manhattan."<br>Travel Suggestions: |
| **Output**      | 1. Rijksmuseum: World-renowned museum... (略)<br>2. Van Gogh Museum: Immerse yourself... (略)<br>3. Stedelijk Museum Amsterdam: Discover modern... (略)<br>*(注：原文输出的是阿姆斯特丹的景点，虽然提示词说的是曼哈顿，这可能是原文的一个错误示例或是为了展示模型的某种行为，但此处忠实翻译原文表5内容，原文输出确实是阿姆斯特丹的景点)* |

为 AI 模型定义角色视角，就像是给它一张蓝图，标明了你所寻求的语气、风格和专注的专业知识，以提高输出的质量、相关性和有效性。

以下是一些我觉得有效的风格供你选择：
对抗性 (Confrontational)、描述性 (Descriptive)、直接 (Direct)、正式 (Formal)、幽默 (Humorous)、有影响力 (Influential)、非正式 (Informal)、鼓舞人心 (Inspirational)、有说服力 (Persuasive)。

让我们修改表 5 的提示，加入**幽默和鼓舞人心**的风格（见表 6）。

**表 6. 带有幽默语气和风格的角色提示示例**

| 项目       | 内容                                                         |
| :--------- | :----------------------------------------------------------- |
| **Prompt** | I want you to act as a travel guide. I will write to you about my location and you will suggest 3 places to visit near me in a **humorous style**.<br>My suggestion: "I am in Manhattan."<br>Travel Suggestions: |
| **Output** | 1. **Behold the Empire State of Mind**: Ascend to the dizzying heights of the Empire State Building... Prepare to feel like King Kong atop the Big Apple, minus the giant ape-sized banana.<br>2. **Get Artsy-Fartsy at MoMA**: Unleash your inner art aficionado... question whether your stick-figure drawings have any artistic merit.<br>3. **Shop ‘Til You Drop on Fifth Avenue**: Indulge in some retail therapy... make your wallet cry. But hey, you’re in Manhattan, so you might as well embrace the fabulousness! |

### 上下文提示 (Contextual prompting)

通过提供上下文提示，你可以帮助确保你的 AI 交互尽可能无缝和高效。模型将能够更快地理解你的请求，并能够生成更准确和相关的响应，如表 7 所示。

**表 7. 上下文提示示例**

| 项目            | 内容                                                         |
| :-------------- | :----------------------------------------------------------- |
| **Goal**        | 为关于复古游戏的博客建议文章                                 |
| **Model**       | gemini-pro                                                   |
| **Temperature** | 1                                                            |
| **Token Limit** | 1024                                                         |
| **Top-K**       | 40                                                           |
| **Top-P**       | 0.8                                                          |
| **Prompt**      | **Context**: You are writing for a blog about retro 80's arcade video games.<br>Suggest 3 topics to write an article about with a few lines of description of what this article should contain. |
| **Output**      | 1. **The Evolution of Arcade Cabinet Design**: ...<br>2. **Blast From The Past: Iconic Arcade Games of The 80’s**: ...<br>3. **The Rise and Retro Revival of Pixel Art**: ... |

## 后退一步提示 (Step-back prompting)

**后退一步 (Step-back)** 提示是一种提高性能的技术，它提示 LLM 首先考虑与手头特定任务相关的一个**一般性问题**，然后将该一般性问题的答案输入到特定任务的后续提示中。这种“后退”允许 LLM 在尝试解决特定问题之前激活相关的背景知识和推理过程。

通过考虑更广泛和潜在的原则，LLM 可以生成更准确和深刻的回答。后退一步提示鼓励 LLM 进行批判性思考，并以新的和创造性的方式应用其知识。它改变了执行任务的最终提示，利用了 LLM 参数中更多的知识，而这些知识在直接提示 LLM 时可能不会发挥作用。

后退一步提示还可以通过关注一般原则而不是具体细节来帮助减少 LLM 响应中的偏见。

让我们看一些例子来理解后退一步提示如何改善结果。首先回顾一个传统提示（表 8），然后再与后退一步提示（表 9）进行比较。

**表 8. 传统提示（用于对比）**

| 项目       | 内容                                                         |
| :--------- | :----------------------------------------------------------- |
| **Goal**   | 为第一人称射击视频游戏的一个关卡写故事情节。                 |
| **Prompt** | Write a one paragraph storyline for a new level of a first-person shooter video game that is challenging and engaging. |
| **Output** | The level begins with the player’s squad being ambushed in a dense urban area... (这是一个比较通用、随机的射击游戏剧情) |

当你将 Temperature 设置为 1 时，你可能会得到各种关于故事情节的创意写作，但它也相当随机和通用。所以让我们**后退一步**：

**表 9. 后退一步提示示例 (Step 1)**

| 项目       | 内容                                                         |
| :--------- | :----------------------------------------------------------- |
| **Prompt** | Based on popular first-person shooter action games, what are 5 fictional key settings that contribute to a challenging and engaging level storyline in a first-person shooter video game? |
| **Output** | 1. **Abandoned Military Base**: ...<br>2. **Cyberpunk City**: ...<br>3. **Alien Spaceship**: ...<br>4. **Zombie-Infested Town**: ...<br>5. **Underwater Research Facility**: ... |

这些主题看起来很适合第一人称视频游戏。让我们回到最初的提示，但这次我们将“后退一步”得到的答案作为**上下文**包含在内，看看它会返回什么。

**表 10. 结合后退一步结果的提示 (Step 2)**

| 项目       | 内容                                                         |
| :--------- | :----------------------------------------------------------- |
| **Prompt** | **Context**: 5 engaging themes for a first person shooter video game:<br>1. Abandoned Military Base...<br>2. Cyberpunk City...<br>3. Alien Spaceship...<br>4. Zombie-Infested Town...<br>5. Underwater Research Facility...<br>Take one of the themes and write a one paragraph storyline for a new level of a first-person shooter video game that is challenging and engaging. |
| **Output** | In the heart of a murky abyss, lies a dilapidated underwater research facility... (基于深海研究设施主题，生成了更具体、生动、逻辑性更强的故事情节) |

这看起来是一个有趣的游戏！通过使用后退一步提示技术，你可以提高提示词的准确性。

## 思维链 (Chain of Thought, CoT)

**思维链 (Chain of Thought, CoT)** 提示是一种通过生成**中间推理步骤**来提高 LLM 推理能力的技术。这有助于 LLM 生成更准确的答案。你可以将其与少样本提示结合使用，以便在回答之前需要推理的更复杂任务上获得更好的结果（零样本思维链在这些任务上是挑战）。

CoT 有很多优点。首先，它投入低但非常有效，并且适用于现成的 (off-the-shelf) LLM（无需微调）。你还可以通过 CoT 提示获得可解释性，因为你可以从 LLM 的响应中学习并查看遵循的推理步骤。如果出现故障，你可以识别它。思维链似乎在不同 LLM 版本之间迁移时提高了鲁棒性。这意味着，如果不使用推理链，你的提示词性能在不同 LLM 之间的漂移会比使用时更小。当然也有缺点，但它们多少是直观的。

LLM 响应包含思维链推理，这意味着更多的输出 Token，这意味着预测成本更高且耗时更长。

为了解释表 11 中的示例，让我们首先尝试创建一个**不使用** CoT 提示的提示词，以展示大语言模型的缺陷。

**表 11. 试图解决数学问题的提示词（失败案例）**

| 项目       | 内容                                                         |
| :--------- | :----------------------------------------------------------- |
| **Prompt** | When I was 3 years old, my partner was 3 times my age. Now, I am 20 years old. How old is my partner? |
| **Output** | 63 years old                                                 |

哎呀。这显然是错误的答案。事实上，LLM 经常在数学任务上挣扎并提供不正确的答案——即使是像两个数字相乘这样简单的任务。这是因为它们是在大量文本上训练的，数学可能需要不同的方法。所以让我们看看中间推理步骤是否会改善输出。

**表 12. 思维链提示示例（零样本）**

| 项目       | 内容                                                         |
| :--------- | :----------------------------------------------------------- |
| **Prompt** | When I was 3 years old, my partner was 3 times my age. Now, I am 20 years old. How old is my partner? **Let's think step by step.** |
| **Output** | 1. Let’s assume my current age is “x” years.<br>2. When I was 3 years old, my age was 3 years.<br>3. My partner’s age at that time was 3 times my age, which means my partner’s age was 3 * 3 = 9 years.<br>4. Now, I am 20 years old, which means my age has increased by 20 - 3 = 17 years since I was 3 years old.<br>5. Since my partner’s age was 9 years when I was 3 years old, and my age has increased by 17 years, my partner’s age must have also increased by 17 years.<br>6. Therefore, my partner’s current age is 9 + 17 = 26 years.<br>So, my partner is 26 years old. |

好了，现在最终答案是正确的。这是因为我们清楚地指示 LLM 解释每个步骤，而不只是返回一个答案。有趣的是，模型将增加的 17 年加了起来。在我的脑海中，我会取我和伴侣之间的年龄差并加上去 (20+(9-3))。让我们帮助模型像我一样思考一下。

表 12 是一个“零样本”思维链的例子。当与单样本或少样本结合使用时，思维链提示可以非常强大，如表 13 所示：

**表 13. 带有单样本的思维链提示示例**

| 项目       | 内容                                                         |
| :--------- | :----------------------------------------------------------- |
| **Prompt** | Q: When my brother was 2 years old, I was double his age. Now I am 40 years old. How old is my brother? Let's think step by step.<br>A: When my brother was 2 years, I was 2 * 2 = 4 years old. That's an age difference of 2 years and I am older. Now I am 40 years old, so my brother is 40 - 2 = 38 years old. The answer is 38.<br><br>Q: When I was 3 years old, my partner was 3 times my age. Now, I am 20 years old. How old is my partner? Let's think step by step.<br>A: |
| **Output** | When I was 3 years old, my partner was 3 * 3 = 9 years old. That’s an age difference of 6 years and my partner is older. Now I am 20 years old, so my partner is 20 + 6 = 26 years old. The answer is 26. |

思维链可用于各种用例。比如代码生成，将请求分解为几个步骤，并将它们映射到特定的代码行。或者用于创建合成数据，当你有一种种子（Seed）如“产品名为 XYZ，编写描述引导模型通过基于给定产品名称的假设”时。通常，任何可以通过“大声说出来”解决的任务都是思维链的良好候选者。如果你能解释解决问题的步骤，请尝试思维链。

请参考托管在 GoogleCloudPlatform Github 存储库中的笔记本，其中包含有关 CoT 提示的更多详细信息。

## 自我一致性 (Self-consistency)

虽然大语言模型在各种 NLP 任务中表现出令人印象深刻的成功，但它们的推理能力通常被视为无法仅通过增加模型规模来克服的局限性。正如我们在前面的思维链部分学到的，模型可以被提示像人类解决问题一样生成推理步骤。然而，CoT 使用简单的“贪婪解码”策略，限制了其有效性。**自我一致性 (Self-consistency)** 结合了采样和多数投票（majority voting），生成多样化的推理路径并选择最一致的答案。它提高了 LLM 生成响应的准确性和连贯性。

自我一致性给出了答案正确的伪概率可能性，但显然成本很高。

它遵循以下步骤：
1.  **生成多样化的推理路径**：多次向 LLM 提供相同的提示。较高的 Temperature 设置鼓励模型生成对问题的不同推理路径和视角。
2.  **从每个生成的响应中提取答案**。
3.  **选择最常见的答案**。

让我们看一个电子邮件分类系统的例子，它将电子邮件分类为 **IMPORTANT**（重要）或 **NOT IMPORTANT**（不重要）。零样本思维链提示将被多次发送给 LLM，看看每次提交后的响应是否不同。注意电子邮件中使用的友好语气、措辞和讽刺。所有这些都可能欺骗 LLM。

**表 14. 自我一致性提示示例**

| 项目                   | 内容                                                         |
| :--------------------- | :----------------------------------------------------------- |
| **Prompt**             | EMAIL:<br>\`\`\`<br>Hi,<br>I have seen you use Wordpress for your website... (邮件内容描述发现了一个Bug，但语气轻松，称自己为Harry the Hacker)<br>\`\`\`<br>Classify the above email as IMPORTANT or NOT IMPORTANT. Let's think step by step and explain why. |
| **Output (Attempt 1)** | **Step 1**: Identify purpose... inform of a bug.<br>**Step 2**: Assess impact... arbitrary JavaScript code execution...<br>**Step 3**: Credibility... unknown...<br>**Conclusion: IMPORTANT** |
| **Output (Attempt 2)** | 1. **Lack of urgency**: ...<br>2. **Non-critical bug report**: ...<br>3. **Lack of personal impact**: ...<br>**Conclusion: NOT IMPORTANT** |
| **Output (Attempt 3)** | **Step 1**: Identify purpose...<br>**Step 2**: Assess impact... serious security risk...<br>**Conclusion: IMPORTANT** |

你可以使用上面的提示，尝试看看它是否返回一致的分类。根据你使用的模型和 Temperature 配置，它可能返回 "IMPORTANT" 或 "NOT IMPORTANT"。

通过生成许多思维链，并选取最常出现的答案 ("IMPORTANT")，我们可以从 LLM 获得更一致正确的答案。这个例子展示了如何使用自我一致性提示，通过考虑多种观点并选择最一致的答案来提高 LLM 响应的准确性。

## 思维树 (Tree of Thoughts, ToT)

现在我们熟悉了思维链和自我一致性提示，让我们回顾一下 **思维树 (ToT)**。它概括了 CoT 提示的概念，因为它允许 LLM 同时探索多条不同的推理路径，而不是仅仅遵循单一的线性思维链。

这种方法使 ToT 特别适合需要探索的复杂任务。它的工作原理是维护一棵思维树，其中每个思维代表一个连贯的语言序列，作为解决问题的中间步骤。模型随后可以通过从树中的不同节点分支出去来探索不同的推理路径。

有一个很棒的笔记本，它基于论文《Large Language Model Guided Tree-of-Thought》更详细地展示了思维树 (ToT)。

## ReAct (推理与行动)

**ReAct (Reason and Act)** 提示是一种范式，使 LLM 能够使用自然语言推理结合外部工具（搜索、代码解释器等）来解决复杂任务，允许 LLM 执行某些行动，例如与外部 API 交互以检索信息，这是迈向代理 (Agent) 建模的第一步。

ReAct 模仿了人类在现实世界中的操作方式，因为我们会进行口头推理并采取行动来获取信息。ReAct 在各种领域的表现优于其他提示词工程方法。

ReAct 提示通过将推理和行动结合到一个**思考-行动 (thought-action)** 循环中来工作。LLM 首先推理问题并生成行动计划。然后它执行计划中的行动并观察结果。LLM 随后使用观察结果更新其推理并生成新的行动计划。这个过程一直持续到 LLM 达成问题的解决方案。

要查看实际效果，你需要编写一些代码。在代码片段 1 中，我使用了 Python 的 langchain 框架，结合 VertexAI 和 google-search-results pip 包。
*(注：需配置 SerpAPI key)*

**代码片段 1：使用 LangChain 和 VertexAI 创建 ReAct Agent**
```python
prompt = "How many kids do the band members of Metallica have?"
# ... 初始化 LLM 和 Tools (serpapi) ...
agent.run(prompt)
```

**代码片段 2：执行结果**
```text
> Entering new AgentExecutor chain...
Metallica has 4 members.
Action: Search
Action Input: How many kids does James Hetfield have?
Observation: three children
Thought: 1/4 Metallica band members have 3 children
Action: Search
... (依次搜索 Lars Ulrich, Kirk Hammett, Robert Trujillo 的孩子数量) ...
Final Answer: 10
```
代码片段 2 发现 Metallica 乐队有四名成员。然后它搜索每个乐队成员以请求孩子的总数并将总数相加。最后，它返回孩子的总数作为最终答案。

ReAct 提示在实践中需要理解你需要不断重新发送先前的提示/响应（并修剪多余生成的内容），以及使用适当的示例/说明设置模型。

## 自动提示工程 (Automatic Prompt Engineering, APE)

此时你可能意识到编写提示词可能很复杂。如果能自动化这个过程（写一个提示词来写提示词）岂不是很好？实际上有一种方法：**自动提示工程 (APE)**。这种方法不仅减轻了人工输入的需要，还增强了模型在各种任务中的表现。

你会提示一个模型生成更多的提示词。评估它们，可能修改好的提示词。然后重复。

例如，你可以使用自动提示工程来帮助训练一个商品 T 恤网店的聊天机器人。我们想找出客户订购乐队商品 T 恤的所有各种表达方式。

1.  编写生成输出变体的提示词。在这个例子中，我使用 gemini-pro 生成 10 条指令。见表 15。

**表 15. 使用 LLM 生成提示词**

| 项目       | 内容                                                         |
| :--------- | :----------------------------------------------------------- |
| **Prompt** | We have a band merchandise t-shirt webshop... Generate 10 variants, with the same semantics but keep the same meaning. |
| **Output** | 1. I’d like to purchase a Metallica t-shirt in size small.<br>2. Can I order a small-sized Metallica t-shirt?<br>... (共10种变体) |

2.  通过基于选定的指标对候选者进行评分来评估所有指令候选者，例如，你可以使用 BLEU 或 ROUGE。
3.  选择评估分数最高的指令候选者。该候选者将是你可以在软件应用程序或聊天机器人中使用的最终提示词。你也可以调整选定的提示词并再次评估。

---

# 代码提示 (Code prompting)

Gemini 主要专注于基于文本的提示，这也包括编写用于返回代码的提示。让我们去 Vertex AI Studio 测试这些提示以查看一些编码示例。

## 编写代码的提示

Gemini 也可以成为开发人员，帮助你用任何选择的编程语言编写代码。作为开发人员，这可以帮助你加快编写代码的过程。

想象一下你机器上有一个文件夹，里面有数百个文件需要重命名。重命名每个文件会花费你很多时间。你知道一点 Bash，可以编写一个脚本来自动化这个过程，但这可能也需要一段时间。所以让我们写一个提示词。

**表 16. 使用 Gemini 编写代码**

| 项目       | 内容                                                         |
| :--------- | :----------------------------------------------------------- |
| **Prompt** | Write a code snippet in Bash, which asks for a folder name. Then it takes the contents of the folder and renames all the files inside by prepending the name draft to the file name. |
| **Output** | \`\`\`bash<br>#!/bin/bash<br># Ask for the folder name<br>echo "Enter the folder name: "<br>read folder_name<br>... (完整的 Bash 脚本) ...<br>\`\`\` |

这对我来说似乎是很好的代码——它甚至有文档注释！然而，由于 LLM 无法推理，并且重复训练数据，因此阅读和测试你的代码至关重要。

## 解释代码的提示

作为开发人员，当你在团队中工作时，你必须阅读别人的代码。Gemini 也可以在这方面为你提供帮助。让我们拿出表 16 中的代码输出，删除注释并要求大语言模型解释发生了什么，见表 17。

**表 17. 解释代码的提示**

| 项目       | 内容                                                         |
| :--------- | :----------------------------------------------------------- |
| **Prompt** | Explain to me the below Bash code:<br>\`\`\`<br>(代码内容)<br>\`\`\` |
| **Output** | This Bash script performs the following tasks:<br>1. **User Input**: ...<br>2. **Folder Existence Check**: ...<br>3. **File Listing**: ...<br>4. **File Renaming**: ...<br>5. **Success Message**: ... |

## 翻译代码的提示

表 16 中的 bash 代码似乎工作正常。但是，如果它能提示我输入文件名，那么这个脚本真的可以重用，理想情况下它应该作为一个带有 UI 的独立应用程序工作。作为起点，Python 是比 Bash 更好的（Web）应用程序语言。LLM 可以帮助将代码从一种语言翻译成另一种语言。见表 18 的例子。

**表 18. 将代码从 Bash 翻译为 Python 的提示**

| 项目       | 内容                                                         |
| :--------- | :----------------------------------------------------------- |
| **Prompt** | Translate the below Bash code to a Python snippet.<br>\`\`\`bash<br>(Bash 代码)<br>\`\`\` |
| **Output** | \`\`\`python<br>import os<br>import shutil<br># Get the folder name...<br>(Python 代码实现)<br>\`\`\` |

> **注意**：在 Vertex AI 的 Language Studio 中提示（Python）代码时，你需要点击“Markdown”按钮。否则你将收到缺少正确行缩进的纯文本，这对于运行 Python 代码很重要。

## 调试和审查代码的提示

让我们手动对表 18 的代码进行一些编辑。它应该提示用户输入文件名前缀，并将此前缀写成大写字符。见代码片段 3，但是真糟糕。它现在返回 Python 错误！

**代码片段 3. 损坏的 Python 脚本**
（使用了 `toUpperCase`，这是 Python 中不存在的方法）

**表 19. 调试和审查 Python 代码的提示**

| 项目       | 内容                                                         |
| :--------- | :----------------------------------------------------------- |
| **Prompt** | The below Python code gives an error:<br>Traceback... NameError: name 'toUpperCase' is not defined<br>Debug what's wrong and explain how I can improve the code.<br>\`\`\`python<br>(损坏的代码)<br>\`\`\` |
| **Output** | The code calls the \`toUpperCase\` function... but that function is not defined. To fix this issue, you can use the \`upper()\` method...<br><br>Additionally, there are a few other improvements...<br>1. Handle file extensions...<br>2. Handle spaces in folder names...<br>3. Use f-string...<br>4. Try...except block...<br><br>(提供了改进后的完整代码) |

太棒了。它不仅告诉我如何解决问题，还发现我的代码有更多错误以及如何解决它们。提示的最后一部分给出了一般改进代码的建议。

## 关于多模态提示？

代码提示仍然使用相同的常规大语言模型。多模态提示是一个单独的关注点，它指的是一种技术，其中你使用多种输入格式来指导大语言模型，而不仅仅是依赖文本。这可以包括文本、图像、音频、代码甚至其他格式的组合，具体取决于模型的能力和手头的任务。

---

# 最佳实践

找到正确的提示词需要反复调试 (tinkering)。Vertex AI 中的 Language Studio 是一个完美的游乐场，可以让你调整提示词，并有能力针对各种模型进行测试。

使用以下最佳实践成为提示词工程专家。

*   **提供示例 (Provide examples)**：最重要的最佳实践是在提示词中提供（单样本/少样本）示例。这非常有效，因为它充当了强大的教学工具。这些示例展示了所需的输出或类似的响应，允许模型从中学习并相应地调整其生成。
*   **设计要简单 (Design with simplicity)**：提示词应简洁、清晰，易于你和模型理解。作为经验法则，如果它对你来说已经很混乱，那么对模型来说也很可能很混乱。尽量不要使用复杂的语言，不要提供不必要的信息。
    *   **改写前**："I am visiting New York... Where should we go?"
    *   **改写后**："**Act as a travel guide**. Describe great places to visit..."
    *   尝试使用描述行动的动词：Act, Analyze, Create, Summarize, Translate, Write 等。
*   **具体说明输出 (Be specific about the output)**：具体的指令能帮助模型关注相关内容。
    *   **DO**：生成一篇 **3 段式**博客文章... 风格应为 **对话式**。
    *   **DO NOT**：生成一篇关于视频游戏机的博客文章。
*   **使用指令胜过约束 (Use Instructions over Constraints)**：
    *   **指令**提供关于期望格式、风格或内容的明确说明（做什么）。
    *   **约束**是限制或边界（不做什么）。
    *   如果可能，使用肯定指令：告诉模型做什么，而不是不做什么。这可以避免混淆。
*   **控制最大 Token 长度 (Control the max token length)**：可以在配置中设置，或在提示词中明确要求（例如："Tweet length message"）。
*   **在提示词中使用变量 (Use variables in prompts)**：使用如 `{city}` 的变量使提示词可重用和动态化。
*   **尝试输入格式和写作风格**：尝试将提示词表述为**问题**、**陈述**或**指令**，看看哪种效果最好。
*   **对于分类任务的少样本提示，混合类别 (Mix up the classes)**：确保少样本示例中的类别顺序是混合的，防止模型过拟合于特定顺序。
*   **适应模型更新 (Adapt to model updates)**：随着模型架构变化，调整提示词以利用新功能。
*   **尝试输出格式 (Experiment with output formats)**：对于非创意任务，尝试让输出返回 **JSON** 或 XML 等结构化格式。这有助于解析并减少幻觉。
*   **与其他提示工程师一起实验**。
*   **CoT 最佳实践**：
    *   将答案放在推理之后。
    *   对于 CoT 提示，将 **Temperature 设置为 0**（因为推理通常指向单一正确答案）。
*   **记录各种提示尝试 (Document the various prompt attempts)**：这是最重要的技巧之一。创建一个表格记录 Name, Goal, Model, Temperature, Prompt, Output 等。这有助于你回顾、调试和迭代。

---

# 总结

本白皮书讨论了提示词工程。我们学习了各种提示技术，例如：
*   零样本提示
*   少样本提示
*   系统提示
*   角色提示
*   上下文提示
*   后退一步提示
*   思维链 (Chain of thought)
*   自我一致性 (Self consistency)
*   思维树 (Tree of thoughts)
*   ReAct

我们甚至研究了如何自动化你的提示词。白皮书随后讨论了生成式 AI 的挑战，如当提示词不足时可能发生的问题。我们最后以如何成为更好的提示词工程师的最佳实践作为结束。

---

## 尾注 (Endnotes)

1. Google, 2023, Gemini by Google.
2. Google, 2024, Gemini for Google Workspace Prompt Guide.
3. Google Cloud, 2023, Introduction to Prompting.
4. Google Cloud, 2023, Text Model Request Body: Top-P & top-K sampling methods.
5. Wei, J., et al., 2023, Zero Shot - Fine Tuned language models are zero shot learners.
6. Google Cloud, 2023, Google Cloud Model Garden.
7. Brown, T., et al., 2023, Few Shot - Language Models are Few Shot learners.
8. Zheng, L., et al., 2023, Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models.
9. Wei, J., et al., 2023, Chain of Thought Prompting.
10. Google Cloud Platform, 2023, Chain of Thought and React.
11. Wang, X., et al., 2023, Self Consistency Improves Chain of Thought reasoning in language models.
12. Yao, S., et al., 2023, Tree of Thoughts: Deliberate Problem Solving with Large Language Models.
13. Yao, S., et al., 2023, ReAct: Synergizing Reasoning and Acting in Language Models.
14. Google Cloud Platform, 2023, Advance Prompting: Chain of Thought and React.
15. Zhou, C., et al., 2023, Automatic Prompt Engineering - Large Language Models are Human-Level Prompt Engineers.