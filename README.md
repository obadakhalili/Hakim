# Hakim - An Arabic Healthcare Conversational Agent

# Approach Research

In our research, we found many approaches to building a conversational agent. All are categorizable into two main types:

## End-to-end model architectures

End-to-end models are one-component systems where the input (the user utterance in our case) is directly fed into the model, and its output (the agent response) is propagated to the user directly.

Our problem requires a sequence-to-sequence model, and a popular choice for seq2seq models is BERT.

![A diagram for e2e models](https://lh3.googleusercontent.com/fife/AAWUweUH2n1UK0kDWQg9xZvl8EXr23f_4gYvqj2vhedoEVGJa1e0ghLkhz4oLyVWLubSeLjxxbxfipD5_GneB_3qg5EbNtdVyaLquRXkzNSj6KGSddK2tUbS9S2hQA_rsQ72bAHHIz05Jvg2bc85kK3FF5cRYU55hMbHhUT3U-Pk-doe0COEmPM2nj_XA5kkogTyzPNSt-hwCqWg5zoIItnnVBBsishaZLcjDWxIE2HJ2ZFnU0lKv5n8byooRHqOMPfXpUYMWzxABNq3OEiT4hLKWSknUkU-UYMxibg6BAB4u3qUiisqGrpjO-Rfy3JQ_3K2AAIAKxsbIHeIeclwhjy5s-PSLfKZn1_RQVEnn4NqvQO1U9VkHfyHrKflU5-oT6YEUFVisf6v7BWme1xgcNuOaCABaXhi1DibC6CBWpCtwpFuxWs0HGpK2FA9pX7NZGBuIIhLGYp6V4Aofh-u0v8TNsqHvVn4BhP4AS4Oi6FpNbb7nvtZ9oPWccspYZ5pOYFatiA9k82SSPtejm4Zw8TCQzirqnYD3LRR700S1FUHhLtE2LKZ1rWZ1HG6lB4ZptetNPrUDL9RlRlZqQN0amBMglToRnz1hgFt6r2y1VbpWWXeiANKyCDAMk9ShX94Cvk2YFLmPKqHmbC7ccAmGizacheGxQXhqWx6SN4GQ1_IA0ebwlb69VIT6ZA59tAnmm11atvf12DAO2Q7gNC8qxOSLE8zP9xX_lSAcyYofqaRWmORflJTrLHKM6mmjOd9mx-_ZYKvJ318aSVLP7rlaMc0_LcMg4h-dBknC3cpLCrzRC01o8egbVx2YXL2s4ewXhqRPdiSPwVXp2vnJ8mShLIONBvUgO2ay_bv4Wfu1EPM2SzTKiy7UQXSIIpIkob7JTjSLaLVoTx9hLMF7vwkol4eqtSM8S_jwh3umjAoyewqBpbHFXbJKjG2jayArzW6p5fyB-Ep-hrRukQvcD6x_82K8JByc99yKmKtUgvYSxdcEbuEsszLlqCMZcAUkRjl7AhIk9jSRBWPPi26dQf2Z-wnxoiDaQ6tfP8qChNYHAuSYTlDrKaK7mB4bQoGskwLFbhBxZmcz63hChODF2ayZZVsHZ1Z_44xNNd2zavgGyFk9ZhbStHXfRhpuN1T-yb9wPGBlcCAQTXr6LS-9etMlbe8ZIZx9Q8n8utTjyBQLlaPkaWtqkRC_RjJl-Exa2BcRIU3bVqsmCRVcnJ2LZM1Ok1TO5MQW53pvxiZjuy5M5vT-6kV6PFeREvO9MnhUK6VtdBaqQZZaagdjEaht2cm8zW_79H3w9qfDvHOylcOdE5GldnAoZFuAPiN7TeqZSB_Q2_tYEPobE9Hpcb9Xm8GV8nEPDTtpsBHqUmLLzJ5c4kHUlnsoNhP0TO8-Q=w1920-h902)

A diagram for e2e models

Problems with this approach:

- Learning a mapping from a user utterance to a bot response is a tough task to learn that requires a huge amount of training data, and it’s especially challenging in our case because:
    - The bot is required to speak in Arabic, which is notorious for its lack of training data.
    - The bot should be able to interact with the users about health-related topics, which is even harder to learn mapping for.
- These types of models are convenient for QA bots, but we need our bot to be able to interact with the user in long conversations and to be able to preserve context.

## Modules-based systems

Modules-based systems are composed of multiple components each responsible for a certain task and organized together as a whole.

![A diagram for modules-based models](https://lh3.googleusercontent.com/fife/AAWUweVlwSIG7ov8EGMAqk3sxJMUCdIQisQ5YdLIykkOE0MjzWsJMvNMm-MRf5uy3SkA2xzscfDD0eia2AdV59cxrafy9ZrtjaZHGTp_giLCFxD3z-CvHv-9YhDxkIfXw-g2dRP3JOS2Tcu4BdQCLCsk8SQb-9tBukGz9BjVZtLmqx5ZPqdK6FNyypxullUX2QWcfH4RD_QXs5-qtB5aaG4jsO9sbY-lX-1NqO73nw1u3ivtnbqGRbRtPQKwlNagi9AKthTBoeTK-Tcw4UjCs1sGEW1CtOQoU85qFulV1IxaPKMeurqDLZDxG1KChHFAGozQ2JsxsQ0PE2fJ0v7ucqIvMwigP4l61jKD3OXp0gBLA38d1EOzLAZYu0XGI82hh3uW52FDBXig9beII0QQ1bod8jCmsSeQLR8sQ8TaB2CoUdhFNmH-sY-YyfUC3kYVMRbKStzqMmzdd2D3yijBXapzRj7Jagzh6RNrCxJwjRPr4HYt0kJc8H8-3oHoED1mFN6T6dvQR9zLWRaZY7WqK14ptmvbPjOXTgz4QYqiBT8O9QuILCt8TZ8fQW1VER8l9ITV8l_GMYMtpWvAeHKaJu14Z22JfmhmtLUmFTjMTvrRMxP6Mk0Mx9zk-ZLTzA17cmy6KgcdQJLAhUB0kBo2KImRb5uE1Q7ugFhSEVHGF-7vza5J8DZ5RFFNu6AiGZjjieQ5FlbiDD4WUQT_GnlgUQ8N235JNWSznVm27v_L2ZiAehHWbidBYliFwaxtrhfeDJ8FJa2-00sEx3_aVGpi_LG3QSGlHzVsEk0cioRnw_szS_W7N-n9oaR3suNR8CnRPfpy0OLhqobJAKZ-K_IBv3HSzKshWG8hzrDYtpuLUf1DVldI6gZwtK54_GFWrxcFIU-wyYhWLQnwUscvIGPXYZa9WQTCj8xYFDiBA6px9DvWmMYRzGdOnG5eIbbqLzYf9RNQItSmtP9f6Tnt2MDrafpgK9uCESJoSWaN0fGCSOj0AXmzTk7vHchMXawbgJUf97a34BEtFcxY9gwXNS31v4q9KqafHO1OmXz2_jPtDzV674YB9bJoANgqh6yxQjLLa7FrbMSerl6q5T-swsEwXqjVqkRVnmuIUy6dMvss-yinKvZMVmjGDMj3QWn08GIjkSO553jBKTY9Loz3ItZl40RDA6iMNu9p-rc-FcIeimTQWe35gWjzth7ofsqqv7ntyMyAu3t0C4Tmu90N-cuGXrfZM_FFEi3EthXYfkB4pbhrHr70JDm6jYaaSXuuWa0nOSJTP7Zroz_rn04wc780FPJnOr4K6ePmua7kJ71WaPHGE0_nbF7QcemaOzaz6Go_SO78HDLDvQBnyh1L_YV8mYvd5P4aWjdctdSqqh68RtW9WWgwWJTbMVjyfw=w1920-h902)

A diagram for modules-based models

There are many ways to build a modules-based conversational agent, and our solution of choice was a Task-oriented Dialogue System.

# Task-oriented Dialogue System

Agents that operate in a dialogue-driven environment (for example, chat applications) and their goal is to accomplish a user task, such as providing a diagnosis given user symptoms.

![Task-oriented dialog system architecture as proposed by its original paper](https://lh3.googleusercontent.com/fife/AAWUweX4YtltXmVEW7cnzSPpKmHgsVMUTalIiHPG7tajc50YPWJUJFzitx7iJ6xcP6_W0z-dSk8OCzLuFa4aTuyKngyu6VojiLWc0NuyNbDaOuTiPluzAQWw_fpcI6SyYjSQnTt4hzvgVqXR2DsOTEZpLZyYzgFglO-wvjkavNDfaykPzYG_LhsOTGN_ttkjTUyYQXdwv6swLpS3-9bNr-PUNQFtgmMBiv_AFjoWT2RVj90rm-hqClTbUNofbcHfnwQ8uVl2F3FzcDc8H8sZq5WbSoaVyzhifP4DzXEfiYj0LsUVB-cOXUT-CAYUm4F5e03USq06-4suJ0-rSDbdlj4cBSJBOnD4AQnmIOhj1w5SfLVJCAemdv5VhsxQrKSM_3fl23g9uQL47FiCycJj-sjs0f4eAfK_vdzgXkLvbL7EnMvCOYZmDJSkBf3tOIS8iHffB-vANdTbeRzzgGMqXX04faMWXsPVOsVzKyb5qfDulN12gkEgkDpA1PSX0An4waPd3lXjAJWZ8a3GIPrhzxYKfLMT81ek3tXWJqHeLoNKneD3MkWY3y0ugv38uteQPiYp9UP8twdGRA3krYHPWlxPRsMqdKgqC8Ds-ZuyOtIkMnRtYiADi-n9sc7c2AzwTqLjFpcKNelT_vZRXipYguVmd9z26zhdod9EtUvql4ipaNFtQ9uq8a9w8ajNYlf53-m9kABbzjrnxcteCMAeky0uNXl60VqfUHlJvPP4OMHI2jRvJmlLwI1P6Xkk54XjmndtsWE8bj97k-of8hnvIHZ0glLnAz-jCTMu9KNJOcK-OWctRUepPXEXuIeCWN2HuPwoY2Pf_4b7gqNJD0gK9iyhEqOyaAhRsSn8zC_S-zdSngtO6CfyRPzyNFL8fvTBRwnoHVa6p84vgIeJzuaR9rYRxy5lSqd7FLA3bd6sxrHo_RKmwnW9MCTI4BD08Z7gDKgitxSBp_VHn3P7CCe0KoRxXsRR7ZbTiu_LUDyYke43Hyl1Xda7FtT0Y1rG1h_wjBCkEez7ijU1QFmFeG8PtE-A5B-xbWJf_zw62t0exnJivXAHUTLZk4WcIfsItwe-4z2Dc3l2-p7wUswkmqPlIofOf4CymTkm8UC00Esxk4kTZ21WFu6IvjUKoDqSN2gL7kKO-xMFwM3y1Erw8gJoQ3dAQZEoZNMwKNGKhUPhu-7mwnajCxYin-WHQK57K6ZEtPiUtAbt2xW27RbHXy870rhxJ3ableYS6RShrn0O8KbGrkvkLYnKmbDpKuChDeKe_76VmD8tuihbzGSCH68MZoOOXouOFv-GYukqd5aYL3AJVOwraYCDA1z3_zUN524jz-aDuPrrT2tRazzVkfZM7JTrACh9SsYnoN5Cga-rXUw-QkarRzJwIdoaaA=w1920-h902)

Task-oriented dialog system architecture as proposed by its original paper

***NOTE**: The components of this system are only explained intuitively below without technical details. That’s because the technology which we used to implement this system, Rasa, doesn’t adapt fully to the system described in the diagram, and so some components might be irrelevant to us (like the user simulator). But we introduced the system here because it will lay out an intuitive foundation for understanding the different components of our bot, and their technical details are explained in the implementation section.*

## Dialogue System

### Natural Language Understanding (NLU)

This component is responsible for transforming the user input into structured information that the computer can use and make assumptions on, called the Semantic Frame. It has two main jobs:

- Intent classification.
- Named-entity extraction.

Example:

- Input: “I’ve stomachache”
- Output:
    
    ```json
    {
    	"intent": "symptoms_report",
    	"symptoms": ["stomachache"]
    }
    ```
    

### Dialogue Manager (DM)

After propagating the semantic frame produced by the NLU component to the DM, it uses this information for the next action prediction. It’s composed of two sub-components:

- State tracker.
    
    This component uses the information passed down to it through the semantic frame, along with information it gets from interacting with a knowledge base (if necessary), to generate the current agent state.
    
- Dialogue policy.
    
    This component predicts the best next action given the current agent state produced by the state tracker, dialogue history, and previous agent actions. This policy is usually learned in a Reinforcement Learning setting.
    

## User Simulator

This component is necessary only during training time to learn the dialogue policy mentioned above.

It encapsulates a certain user goal, for example, knowing the diagnosis of a certain disease given user symptoms, and it interacts with the dialogue system component to teach the policy optimal action prediction.

# Implementation

Our technology of choice for implementing the described Task-oriented system is Rasa.

Rasa is an open-source machine learning framework for building conversational agents. It provides rich APIs which can be used to build various task-oriented dialogue systems.

![Rasa bot architecture](https://rasa.com/docs/rasa/img/architecture.png)

Rasa bot architecture

## Agent

The interface of the bot. It has APIs to train a model, load it, and receive and send messages through its RESTful API endpoints. It wraps around the NLU and DM components and uses them for the actual user message processing.

## NLU Pipeline

A series of steps that are responsible for intent classification and entity extraction training and prediction. Those steps are defined in the `config.yml` file.

The intents and entities are defined in a `domain.yml` file, the file which represents everything the agent knows (intents, entities, slots, actions, and responses). The related code from the file:

```yaml
intents:
  - affirm # e.g: Yes, indeed.
  - age_report # e.g: I'm 21 years old.
  - deny # e.g: No, not really.
  - dont_know # e.g: I'm not sure.
  - goodbye # e.g: Bye. see you.
  - greet # e.g: Hi.
  - nlu_fallback # Any message that doesn't fall into one of the other intents.
  - observations_report # e.g: I have a very bad stomachache.
  - restart # e.g: I would like to restart this conversation.
  - sex_report # e.g: I'm a male.
  - symptoms_inquiry # e.g: What are the symptoms of COVID-19.
entities:
  - sex # Synonyms like (رجل, ذكر, انثى, ابي) are mapped to either male or female.
  - age
```

The intents and entities’ training examples are defined in a `nlu.yml` file. A sample from the file:

```yaml
nlu:
- intent: deny
  examples: |
    - لا
- intent: affirm
  examples: |
    - صحيح
- intent: dont_know
  examples: |
    - لا اعرف
- intent: age_report
  examples: |
    - عمري [15](age) عام
- intent: sex_report
  examples: |
    - انا [انثى]{"entity": "sex", "value": "female"}
- intent: observations_report
  examples: |
    - أشعر بألم في اعصاب يدي
- intent: symptoms_inquiry
  examples: |
    - ما هي اعراض مرض السكري؟
- intent: greet
  examples: |
    - مرحبا
- intent: goodbye
  examples: |
    - وداعا
- intent: restart
  examples: |
    - اعادة البدء
- intent: nlu_fallback
  examples: |
    - كم عمرك؟
```

Entity tagging is done by Rasa using the BILOU tagging scheme:

![https://lh3.googleusercontent.com/fife/AAWUweWF48fkQohmrjPIqwdtu1herMCrPhR9p_RoRh7PdB5a0Pdotzsdsxl64OjESOxuVMdQuPFOYh2pUb72CCmH8RC4g4UVkAlQxd_lxwExPoOFZm7cpvDkXZ9IblTutyLhW6B05sRulTtijdeiiBu0yIpJSzQy5peCXT78aIJwGzxEZWxCZE5A-ku2Ek9XEhDlNCAvzCH6DYE41gRAXNhly0JY1FZUleAJ21JuCUA9IqlbcY-KC9YyB5bh8Z7Ocx1IgMg8pTqOyypqVL_nQyqaFohOFlgmT53DaDTNRP5IXGbDQHtFyC1GsxeCtbi0AiNGaQ_GobpCjJlT4k-XlVRS8D2p9HkStgpB5EgBEDMMI1kajvQRh9NfgmvFfLr_9QSemqiyy704zHQ0N5Fzq-hPaVEM66-TZpvRHNiZYZuKygNwEgjNzeX0a03DPc5qq9KFuxY_y5n2JoF4nn5ho8vfXD0iYvoLi3Y_QR2AwDsrt7fgpmJ5Ko9Dh6WI2s5ZCvvYXFk7BU7-6fbx-nxRRGLGtsrIXdotR24XadcfJBormWh0TpIp-buYKo5XwfOj42k2PlJp5OQeUi1cntoD_LqP04eHoFnDO-MXZAJKKxYCQQeha8sAmW4kYDg1wzPrKnMxtEYoRBl1_exYZXECwLMThJu1t99SR28GkvZvlxvLZT4PRjmmQ3nBstkN2jmqHBPGe-yE3g4Bo7pppt1A2brsp8IxTTvhT4MkjZErXq_Tai0LDrwAACFIhbFT7y3gAd2sm9HV5atpDTm0htA9HZvFvwwd33JpaFvHtPxpwJlbpNKCOqAcy6tNlHDmvVXnVyqV_RfKKTs1u5iosTUvtTI2CnfYYSngvRz8nIZqNFTGZjz6MKzCa-H6ZJ_fP5DoWWicw6SUBWsISIXUHtNIWzyaQAVPfm_2Cy1TFsD_u4aMY66BfPiLZ8eq4kOpTDkPO0eNjvLKZExp_Bqx-QNDa3l7OnVjFYH28CP_0njOZcu3tJBzK3n--bkVUxc00bxP019U9tdT30MN_fzB4zUT-BloCywfuVQ2U_Mn7QL7GlgLEPOJ8434k6cGs9h3Q0or2SIqWndXWy_OMLjS200BUGzq9XzkoxOdUKFHa4anRWncKE6d1GKIoTqZd0XFC0MIoWZKRT4nYbGc3EnmXBEIRWbGGTeLb1iXu0rmdQ9KfjAb1cx7Q-M5XEZnVHKHTbF4LTzHMcbPcvx48d22lTjVyBZqDj7aXGC8WjyrZ1gcFEpokVAIMBCPNyhJ4v7W0pTLV-KXCYGbDJYecSLhAmWv-nmT-DKq3LcgI8rndfUdGvCxS7ZtEHF6LKryXv1wXI75VUMTxh7VEarMhM_CnUrHseMeOSDlBaRhFE2lT-90AXlIoYysFC0ZZ30WyQ=w1920-h902](https://lh3.googleusercontent.com/fife/AAWUweWF48fkQohmrjPIqwdtu1herMCrPhR9p_RoRh7PdB5a0Pdotzsdsxl64OjESOxuVMdQuPFOYh2pUb72CCmH8RC4g4UVkAlQxd_lxwExPoOFZm7cpvDkXZ9IblTutyLhW6B05sRulTtijdeiiBu0yIpJSzQy5peCXT78aIJwGzxEZWxCZE5A-ku2Ek9XEhDlNCAvzCH6DYE41gRAXNhly0JY1FZUleAJ21JuCUA9IqlbcY-KC9YyB5bh8Z7Ocx1IgMg8pTqOyypqVL_nQyqaFohOFlgmT53DaDTNRP5IXGbDQHtFyC1GsxeCtbi0AiNGaQ_GobpCjJlT4k-XlVRS8D2p9HkStgpB5EgBEDMMI1kajvQRh9NfgmvFfLr_9QSemqiyy704zHQ0N5Fzq-hPaVEM66-TZpvRHNiZYZuKygNwEgjNzeX0a03DPc5qq9KFuxY_y5n2JoF4nn5ho8vfXD0iYvoLi3Y_QR2AwDsrt7fgpmJ5Ko9Dh6WI2s5ZCvvYXFk7BU7-6fbx-nxRRGLGtsrIXdotR24XadcfJBormWh0TpIp-buYKo5XwfOj42k2PlJp5OQeUi1cntoD_LqP04eHoFnDO-MXZAJKKxYCQQeha8sAmW4kYDg1wzPrKnMxtEYoRBl1_exYZXECwLMThJu1t99SR28GkvZvlxvLZT4PRjmmQ3nBstkN2jmqHBPGe-yE3g4Bo7pppt1A2brsp8IxTTvhT4MkjZErXq_Tai0LDrwAACFIhbFT7y3gAd2sm9HV5atpDTm0htA9HZvFvwwd33JpaFvHtPxpwJlbpNKCOqAcy6tNlHDmvVXnVyqV_RfKKTs1u5iosTUvtTI2CnfYYSngvRz8nIZqNFTGZjz6MKzCa-H6ZJ_fP5DoWWicw6SUBWsISIXUHtNIWzyaQAVPfm_2Cy1TFsD_u4aMY66BfPiLZ8eq4kOpTDkPO0eNjvLKZExp_Bqx-QNDa3l7OnVjFYH28CP_0njOZcu3tJBzK3n--bkVUxc00bxP019U9tdT30MN_fzB4zUT-BloCywfuVQ2U_Mn7QL7GlgLEPOJ8434k6cGs9h3Q0or2SIqWndXWy_OMLjS200BUGzq9XzkoxOdUKFHa4anRWncKE6d1GKIoTqZd0XFC0MIoWZKRT4nYbGc3EnmXBEIRWbGGTeLb1iXu0rmdQ9KfjAb1cx7Q-M5XEZnVHKHTbF4LTzHMcbPcvx48d22lTjVyBZqDj7aXGC8WjyrZ1gcFEpokVAIMBCPNyhJ4v7W0pTLV-KXCYGbDJYecSLhAmWv-nmT-DKq3LcgI8rndfUdGvCxS7ZtEHF6LKryXv1wXI75VUMTxh7VEarMhM_CnUrHseMeOSDlBaRhFE2lT-90AXlIoYysFC0ZZ30WyQ=w1920-h902)

Where:

- U (unit): Marks single-token entities.
- B (beginning): Marks the beginning of multi-token entities.
- I (inside): Marks the inside of multi-token entities.
- L (last): Marks the last token of a multi-token entity.
- O (outside): Marks non-entity tokens.

The training examples were either scrapped from medical forums like [WebTeb](https://www.webteb.com/), or written manually by us.

## Dialogue Policies

Rasa has multiple rule-based and machine-learning policies that can be used to decide what action to take next given a user utterance. The desired policies are configured inside the `config.yml` file.

At every conversation turn (initiated by a user utterance), each of the defined policies predicts the next action to take by the agent along with a confidence level, and the agent predicts the action with the highest confidence. If two or more policies predicted actions with the same confidence level, the agent predicts the action of the policy with the highest predefined priority. And if two or more policies of the same priority predicted actions with the same confidence level, the agent predicts one of the actions at random.

Actions predictions can either be:

- predefined responses which are defined in the `domain.yml` file as follows:

```yaml
responses:
  utter_confirm_restart:
    - text: هل انت متأكد انك تريد الاعادة من البداية؟
  utter_greet_back:
    - text: مرحبا, انا حكيم, طبيبك الالي 👨‍⚕️.
			يمكنك ان تبدأ بان تخبرني بأية اعراض تشعر بها, وسوف احاول مساعدتك من هناك
  utter_goodbye:
    - text: وداعا واتمنى ان اكون قد افدتك بمعرفتي
  utter_observations_too_long:
    - text: الرجاء التأكد من ان طول الرسالة لا يتجاوز 2048 حرف
  utter_specify_sex:
    - text: ما هو جنسك؟
  utter_specify_age:
    - text: ما هو عمرك؟
  utter_no_observations:
    - text: لم يتم تحديد اي اعراض في الرسالة, الرجاء التأكد من محتوى الرسالة
  utter_pediatrics_not_supported:
    - text: للأسف انا لا ادعم طب الاطفال تحت سن 13 عاما
  utter_age_too_high:
    - text: عمرك اكبر من العمر المحدد وهو 130 الرجاء التأكد من العمر المدخل
  utter_symptoms_inquiry_out_of_scope:
    - text: حاليا انا لا ادعم الاسئله عن اعراض الامراض
  utter_fallback_message:
    - text: اعتذر لكنني لا افهم ما تحاول السؤال عنه
  utter_default:
    - text: هذا السؤال خارج نطاق معرفتي, هل انت متأكد من صياغة السؤال
```

Responses use cases:

- `utter_confirm_restart`: Used for when the user asks to restart the state of the conversation.
- `utter_greet_back`: Used for welcoming users.
- `utter_goodbye`: Used for farewell messages.
- `utter_observations_too_long`: Used for when the message reporting the user observations is longer than 2048 characters.
- `utter_specify_sex`: Used for when the user sex is not specified.
- `utter_specify_age`: Used for when the user age is not specified.
- `utter_no_observations`: Used for when the previous user message that was identified as reporting observations doesn’t have any actual observations. e.g: “تؤلمني كثيرا”.
- `utter_pediatrics_not_supported`: Used for when the specified age is less than 13 years old.
- `utter_age_too_high`: Used for when the specified age is more than 130 years old.
- `utter_symptoms_inquiry_out_of_scope`: Used when the user asks about the symptoms of a disease. We indicate that we understood the request but it's currently out of scope (its skill is not implemented yet).
- `utter_fallback_message`: Used for when the previous message doesn't fall into any of the predefined intents.
- `utter_default`: Used for when the dialogue policies fail to predict the best next action.
- or custom actions, which are carried out to the actions server to be executed.
    
    Custom actions can execute any code, most notably, they are used for integrations with third-party services and knowledge bases. And this is the case for us as we integrate with Infermedica as our medical knowledge base through custom actions.
    
    Our custom actions defined in the `domain.yml` file are as follows:
    
    ```yaml
    actions:
      - action_setup_interview
      - action_diagnose
    ```
    
    - `action_setup_interview`
        
        This is the action responsible for setting up the diagnosis interview. More specifically, given an `observations_report` message, it:
        
        - Translates the message into English using Google Translation API.
        - The English message is parsed using Infermedica’s parse endpoint to get all the observations (symptoms and risk factors) mentioned in the message.
        - The observations are stored inside the `collected_observations` slot.
            
            Slots are the bot's memory, stored as a dictionary of key-value pairs, and defined inside of the `domain.yml` file.
            
            The observations stored at this stage are marked as `initial`. That’s because when using the diagnosis endpoint of the Infermedica API, observations that were reported initially affect the quality and speed of the diagnosis significantly.
            
        - We use Infermedica’s suggest endpoint to get observations reported by users with similar initial observations, sex, and age. Those observations are then stored inside of the `observations_questions` slot, and the user is asked about them later in the `action_diagnose` action.
            
            Observations returned by Infermedica’s suggest endpoint are marked as `suggest`. That’s because they help the diagnosis process just as much as the observations marked with `initial` do.
            
        
        ![action_setup_interview execution flow](https://lh3.googleusercontent.com/fife/AAWUweUa8DlhTeom1UGXcWsjqHWR7UgbGZfugUU1gz8hFDErUAG00OnJoYNDramTepJPJUFW3HIm36jc2RbuS1ifsAJIyIsApbVZ3IFbO9Ukb7HI1Ef_G0Zq9Lt3ms7kLtj0MfXyOXm9Stg4QvDE0Ybe5_fiRSsGn5VFqsMzj-2jcj2sZbR_XYZNrQXlnAF-NMnymbBQd8A9yDHx37nhsTAZbUUMNh9-CPwmH4a0w29bjATU-tabVyNcd0xKQgSwC_EYf3kB-LNFxaijDPWomlkJGFXhA8QAC7lMk6O56gEn6XqrPJ1qbbTZQvWli4fleRXHKCyVcrDrcy-M0z2dPZqFvQ520wSz81RO5BagG3LpYhgtrMadSj17n9AfQ7Xh0jn4rKIU3WVwFwsuxivhGzRc66AhVZdjkRXoNLmYnGLMXh1gi-meUf1H87y5O1JXzFf6zyPGlbT8ctGK39z30GdKFinIkqomXpZpr8KKLlYgnJ7ZiQ8jixtLQJcqt-ng5esr93P4Opk7_LWNnOZmsjZqNwJ3jUw-tWv582F_6s_OjZQiNdSLv8F5dAYxPfvGMiGZg6c_fGfPD8GZyqBoPbAELz8P0bzlVQIQeoBrH_dci68976nFJ5JnxJKE0UiTFgnqvp0yjy7Bj6Len73_6KtcKHNJleM4LCGVJvrgjhjogvQpL1wCVQhjt6pe0a57Xa3dCDlB4mPYCk6VkRtR1gN4s91unItPDGNP2e1l8zXtNEx2a41vW63cz-xPxi_k640Fo7ywbQHRcvUohCnWJTScCYjb_b-t4iBGYU22FAk5SuU2tqLR2Scb8rKqUnl53LJHuOVpfxw5UpCWvR_Xq6kJtXDvr9jP_f24osyO_RzNrbFyksI9rl9h4Dv_xI68agzlOVlRRmDWKrpw9L0dobR4zIMXU8j9y0NKRjWfykZlwKaax-Aui93vsnN7VexDsQzpDrqGpahPS9wQl3lpS4gZRR_P-06loossnZtr6HXHergvCXKJwJWOUgZmxOBLW33--1HWO4RI6YV2xZqagkJy4UeKXTgdokI11T_Ylke3ueGHD3ATjrJtS0IcjlBQe5HsGvv_092_fHdt74Wguh_3chSGDtnL3Sl68CkziDiNxb7tBEcbxZD55eJqnG3hNP7I2Wstyh49Z0Vbmwzz9NyvvQVKUcBjrmrCjgYOXBHeFqnWadmdyBiAbX_L7Tl3qjP1rnOLNwbkyOYbFYA-Ugx0yxR650qxhm-Oi2o7BTl5y8Toh0d7aKrQtgNHwFlp4FDJPgTdo6dhYralLXJVkge18CQB3NL2cAdNrj76ENH0IjtHDyEe_jbR9sMzYnwGhzfkKwnU3ZC3iWuTwdhLf4vk1zd2_WPYNh0eFJEzsQNtKzSyU7tmJlK-7w=w1920-h902)
        
        action_setup_interview execution flow
        
    - `action_diagnose`
        
        If the previous action was executed flawlessly, action `action_diagnose` is executed as follows:
        
        1. Ask the user about the state of the first observation stored inside the `observations_questions` report.
        2. When the user replies with `affirm`, `deny`, or `dont_know`, store that observation and its state inside the `collected_observations` slot, and remove it from inside of the `observations_questions` slot.
        3. Repeat from step 1 until there are no more observations inside the `observations_questions` slot, and then invoke Infermedica’s diagnose endpoint.
        4. If the result of the previous endpoint suggest stopping the interview, we stop it and output the diagnosis returned by Infermedica. Otherwise, we add the next observation question returned by the previous endpoint to the `observations_questions` slot.
        5. Ask the user about the state of the current (and only) observation stored inside the `observations_questions` report.
        6. Repeat from step 2.
        
        ![action_diagnose execution flow](https://lh3.googleusercontent.com/fife/AAWUweV9hB2WpzdoUacD8i4F3qC6fyIeNVNE7BOhmOYwtIpZeePtfjqGDxA0LHT8r-9_OvVHyEpyrY6IPpXXMmBaaEAO1Y3VbjlCOAsBIt795g2ZFwW9P9XPVm-bgRnj6q4m9S6idMg-srYa4_i3l_6FjutNPvzqfk8Ocq9IS6aVQT8vsVbCl9ccuEywYxQAsSSyt4MIoClJsmdglVv65aC-Xb002wWi_RqxE_cvCXObn8Z8Sdm1__qeXSSTqSkep3cu69yu94dAw4Hlt7ex8jxsmJkKhQ46ssWdCpSnZmRUjYt_5uR0rOJWTgtn0tWUNF1u-fSs4Rn8fm1H9DrK9VwZWaeMZ7-b5X2FeERt2xwAAiCD_JKmsIFKinHgTz-95mUaF96GizCEHmIBKpnmTo5VVamp5taCHq79eZEEdbnWETQjNK06TsyllbqUEqYAHlGFNrGlKhMAXXD-6e38UXIqkyWkVRr_X5Na5E-1fy1T8s0NUw87euGDay-P89gRf1LdVsVUKrmmta1-A3_Jibay6jiRh9omumzlW-3LNb2H7ozF6saQbFwQKp_mdmUqnTILcrANLZwt_7GYja66AB-3nX1NU9cbvIF5nhXYiKg4OZ0x6CRoRJ3v8IBzvSN2gmIecP8kP63ZkgnpVz67XcDFVvaYUkLeWqVhUCoy--fJoK4WkL4aQx2JlyRQY09ucr0IDA8Nt7IkiHHMoL1ZODxnuk12vQSGg8tFpZM7NSib7b0EuB-h2KMRddkcpxrqNTR5bVzUKac8Mo8_EFI4nkSi0b9jUxOq9eyjro6dAr0LidPbdiI5NFovFGbpE2MhKQ2xAFWOzMAbPf1LEqvE4Na3kz8PvFEC5RTlfKBWSW0MVaiv9XddDY6D2dyeydO95NO7ozEIguorQp9_D8mxco1XPuKcfXC-uSk2cq6WXXAUxmtEgTK1xGclLcSmY4trOz6ItETqX0nwQ9HQ2XlDOBprp7BOaLLtVb-NPS0kif8iQ13QwiWe_BM7e6-UBUm_K94ycYIGhiIxG5NZ4x-BKhWOexV78wACbLHue8jsSvxlr51vfZpT_6Iy8ovqgCNhWemH6Q_0GRlW6VGxk1Vfw-gbJ0lbHLUFNrtYEPyL2U5DIJJ1dg-ZTrFwZ6HA8OutKk-X2mmtnMx33PTZPdQWMPqB9FaJalI0nfL-G49z5v5W3Q9bV1hdc3MhPh7B5COWXmN4gdl8U_2BqGjGNpz57O2f9kPVNO-ZdTgplarB3FRsAkiv0yMuURA8ewYjR5pmO_q9VoHCZzVHa0zKxWTiLYkTpkiKvnmDUEp9waY_qMYJwo_imTd-mKZBLwhPeWl5pMacWi7sReWDxc9EmGF_qb92jYn9uoaI4j6c0ikVlyWos5kb70PrpK63Rg=w1920-h902)
        
        action_diagnose execution flow
        

Training data format for the dialogue policies:

- Stories, defined in `stories.yml` file.
    
    A representation of a conversation between the user and the bot. In this format, user inputs are expressed as intents (and entities when necessary), while the bot's responses are expressed as action names. Example story:
    
    ```yaml
    stories:
    - story: diagnosis - happy path  # The name of the story
      steps:
      - intent: greet # The intent of the user message
      - action: utter_greet_back # The bot's action
      - intent: observations_report # A user message that's reporting observations. e.g: I have stomachache
        entities:
          - sex: male
          - age: 15
    	- slot_was_set: # Setting the sex slot from the detected entities
        - sex: male
      - slot_was_set: # Setting the age slot from the detected entities
        - age: 15
      - action: action_setup_interview # Calling the action_setup_interview custom action
      - slot_was_set: # In this action, we store the initial observations detected from the observations_report message
        - collected_observations:
          - id: s_100
            state: present
            source: initial
      - slot_was_set: # Also in this action, we store the observations to suggest that we got back from Infermedica's suggest endpoint given the user initial observations, sex, and age
        - observations_questions:
          - id: s_81
            source: suggest
      - action: action_diagnose # Calling the action_diagnose custom action to start the diagnosis interview
    ```
    
- Rules, define in `rules.yml` file.
    
    They have the same format as stories and are used to describe short pieces of conversations that should always follow the same path. Example:
    
    ```yaml
    rules:
    - rule: greet back whenever the user sends a message with the intent `greet`
      steps:
      - intent: greet
      - action: utter_greet_back
    ```
    

## Actions Server

When the agent predicts the next action to be a custom action, this custom action is invoked by calling a RESTful API endpoint that follows a Rasa predefined standard for communicating the input and the output of that custom action back to the agent.

![Illustration showing the nature of the relationship between the bot server and the custom actions server](https://lh3.googleusercontent.com/fife/AAWUweW3rgeZ-TGoM9eL1wrKCg7zjdMM1jM5QmLJ5yhIjXkIC3L1nHnf7jYpePvk98ZLG7h60MXheek8LiiW6jk4wP9JyhPoThShVWs_NvSodV-BW0hX_Gerb4aXP0irm4KWZS_NDyGSXHXzdndGSbVsMFbdAzLl-gmpk-x8FYRMS0AHLpQRG3bQE0M-12cN3rqfh2P4_xzu2sGzcVyxw2rC1OK3RA6G4RUE5Kz42P3tXbv3DdezWI18TqD1liGKzXNOlcmuSwn2mMvasXzmjh2VEAXmw0qXm2F6Bc467S7bs702weGjwLY9G8ejU99fDddK4X3-rAXl5Sd922A8Vg_3msokBycdNG4ncKMXuAQw_ixMLqrlkZZ0fmc3XYRdC6w_LbIw8lmiKLgFEL8_gvgivw7T9XbxqUHld698FLxfIavNVVqSDgbyE-5MEJgMEOWF2ARIQSmx5q6e4C2x7LfbsZjQh3dOfMHLXp0alc7DhlfX74U8JobN9w6OZiQ-9aeTzzOlJHGDe66MlaHp6b6Uw2KrijsjAy9sNGPf20C_xT21f5GeiJM0rxMbHaXhBI5EberM-054W7Xwhx-FzcF1GwAoZmO6tzb6tVUAS3B6wGAu-h3mQzoAunI88UNQXrhw5iVawykLAKNrWIPAl5G7yoQQiRG5fh4MmESI-eBzZ1t8qfRMGwXxXtg6mlJGEiRDduDjNqm2MWri1Xe1Bia7IoJF7aX1AVUzknuuRsE-j9s4rHfcTTdaH1D8Yq_yQSZkLBehWRn_LQCtByRytsgL4bUmWox6u-4viT55bcQ3RHuOPkkAD3PSJHeSlLRX196xE_RTwso2J0eCkTXWqMrz2WI_3f_5DRCWqi-vDqQbOQyOMtVyLpHDRBlYsKaEi5Hd1LRO9vdgxG0KktVYHs_ji8_5Ro4E_P1JDiZAlwoHPZhNNUOO23Y6tPMLJcn0v4f7Y4BN3aMSkmJutgOwz-rocP1YJMwo0OoREQ9OuVRv3RnZoBDzqsSjUhJCLeZXpWPVq97UE0cPA1rA5mRC9zejC-KeuiBVVwpmAJ5b5OTxNePvzeTrP76nHI1t3X6gZRkCh_F9CX6p8cKoKDHLY5sB0zxt66OozFIBS-tUS1GdyC7PI3i_bB79pQkSMiEfXom0s9XPHAWmOKe4V8oLfd8FAM3NLNnJ8Ua37Dpj4K1MKOp1rpOB4oeHvQANqPtMdsQ27cw36fZ0sa1hJV8vnxIEhm6fYWQ51Y_EFhHFH_XZ_WakTf8W3Wd_wof1_cGMamMrxBPNAV1FwiM_apG8RNN7BF73u8wVkCpmeA0-TOXkD1UqXEj4FdSbDpiIGunSDs44N_Un7Pp-DY1vKRokIV5ZxLuT7ZhLKmssA8xbRIuuiV_v0mbL5wxGsA=w1920-h902)

Illustration showing the nature of the relationship between the bot server and the custom actions server

The endpoint of the actions server which the agent has to communicate with is defined inside the `endpoints.yml` file.

```yaml
action_endpoint:
  url: "http://localhost:5055/webhook" # When the actions server is hosted locally
```

## Channel Connector

A channel connector is the means through which the agent receives user messages.

We can integrate it with our own website, Facebook Messenger, Slack, Telegram, and many other channel connectors. We choose Facebook Messenger as our channel connector of choice.

## Tracker Store

This is the place where the bot’s conversations are stored. Rasa provides out-of-box integrations with different store types like SQL, Redis, and MongoDB. But for this phase, we used the default in-memory store, which stores the conversations in the server’s memory.

## The NLU Pipeline and Dialogue Policies Configuration and Technical Details

### NLU Pipeline

The configuration of our NLU Pipeline inside the `config.yml` file:

```yaml
pipeline:
  - name: WhitespaceTokenizer
  - name: LanguageModelFeaturizer
    model_name: bert
    model_weights: asafaya/bert-base-arabic
  - name: DIETClassifier
    epochs: 100
  - name: EntitySynonymMapper
  - name: ResponseSelector
    epochs: 100
  - name: FallbackClassifier
    threshold: 0.3
    ambiguity_threshold: 0.1
```

- `WhitespaceTokenizer`
    
    This component is responsible for splitting up the input text into tokens based on the whitespaces between words in the sentence.
    
- `LanguageModelFeaturizer`
    
    Uses BERT with its weights downloaded from HuggingFace to map each token from the previous component into pre-trained word2vec embeddings.
    
- `DIETClassifier`
    
    A multi-task architecture for intent classification and entity extraction.
    
    Takes as input a features vector for each token, and thus it requires the pre-trained embeddings from the previous component.
    
- `EntitySynonymMapper`
    
    Normalizes the entities extracted by `DIETClassifier` into male or female.
    
- `FallbackClassifier`
    
    This component classifies a user message with the intent `nlu_fallback`  if none of the classified intents have a confidence level higher than the defined `threshold`, or if the difference between the highest confidence levels is greater than the defined `ambiguity_threshold`.
    

![https://lh3.googleusercontent.com/fife/AAWUweWDvkYmFrM-Jd9-t1KdftbvFnLjmD77ZHUauJCZFI42imQ7xUw43H1gZNux0DImbL5Hdpdcv2l6eYJeC8eYcyVDOmTLb2GKP2ovVhH3x7OxHwwUtYnHwrug4P3ZW9l6MkEcZ7G46h4QCM80YooxzUwY5jbLoOZ07wOhcN6bYWKXP-1ZmOV4iApaCvFnow5CL0QsvuyhRf_7rMnhTrEVNazViVHsXT3P4E4XvVr6rZf3uLlnjRf7NyHWbrOvc0Ob7Dd7aZG5u77lInD3nuTQqfshQfH0HGMMBEzrONRRd7OPQ7Purc_hvG5nwgSbU7Mfm6F2-8cmvGKrphnW3zvg9e4z0nuAQkH1mS8cwV6hp1EER6Cm5_ym_axgh3gHX1ETKDbiQ5ddhQoRA5JOry1WpmbSBg9POPveUUAtF2yvTmwhk_lkTA4pUZi85gp5g--GlW_LPa01VvVJqIim4Lb1LqQA-eAMnYiNh1oaxUw238d5_lLOnwKd_MObNjm7_JRUpwcwYUZ0IilyhME33uTwgHjFe0lrdsEZxwKzbvyspjse36k5hqCX_tjwscrZq7TndBkaHJET4DsC5SPMV24ptNkJDLLTjX5TFZrqctXwbcbyxRfWFvegGDcHkoZkDvbYV-3gW0itod_5630LUgvPMZBKl4UK2EsdEB4CiaALNwVuFXV-e0vQXIDM8daIQ_PyUXxtNDs0nDE0VF8melfRpnkIjD-NpxbLbTmnPuhvlB9aa8wngrZuYy9bVvUDLkDiXspqlHTpGYCk5NTIYDveeY-GaIQhTJUTbLPghenpWL8qaYkHkokumWOBlOzapAJXMl5O5HwCB8Q8M_XjnJmtfHYj3SkMznOp4EfHCtj4796vIkW2kEKMvfak1MFD0OBqwmYIg12oMYa83QmoLyGYLTq_oc8B4E5amIZ3lk5zpKPXoBAq1B9UULSutlhKm1I78pytcM9wAJhuD5TdrS1RtcdzJP3s2-tBiPwhR5Ubk_BCuaQ30qEMLzKJX0i5E9KE6kHA8nkY6xQbeFUYFVm4ZOHMoZ_KdNmAA8dWKIHZgL3pfjJMQN4GSWhhq4NyRe2AFaw0afmiM5SIcilWhKpKAY_7TOHmBxHNcqw2dmaUALSyqBVuQvd8y368XPUiIDmOpIQ7juYdD5oXoUXT_8WRRgiIGJ1p8ZWS4l2LmSbiVTGwZ61N0-QxFgjIRGrTa9-7O6kEqiO79Zc7dTYJYJxcCXfbTuZ42N-pWsfdOizCOe0PJ-PV4bjsxAD6W7JC3yVt8EVINrreYC_B4CKh3e6XGbQ_pY76m1Eaa4wMPgVcx6gLU5syNnguJKqwS3vGP8Bxc3ZBifodzNw2QODv8H-e_g7DKrH5W1Y70Hcr5_hmjABi2vryEFUKng=w1920-h902](https://lh3.googleusercontent.com/fife/AAWUweWDvkYmFrM-Jd9-t1KdftbvFnLjmD77ZHUauJCZFI42imQ7xUw43H1gZNux0DImbL5Hdpdcv2l6eYJeC8eYcyVDOmTLb2GKP2ovVhH3x7OxHwwUtYnHwrug4P3ZW9l6MkEcZ7G46h4QCM80YooxzUwY5jbLoOZ07wOhcN6bYWKXP-1ZmOV4iApaCvFnow5CL0QsvuyhRf_7rMnhTrEVNazViVHsXT3P4E4XvVr6rZf3uLlnjRf7NyHWbrOvc0Ob7Dd7aZG5u77lInD3nuTQqfshQfH0HGMMBEzrONRRd7OPQ7Purc_hvG5nwgSbU7Mfm6F2-8cmvGKrphnW3zvg9e4z0nuAQkH1mS8cwV6hp1EER6Cm5_ym_axgh3gHX1ETKDbiQ5ddhQoRA5JOry1WpmbSBg9POPveUUAtF2yvTmwhk_lkTA4pUZi85gp5g--GlW_LPa01VvVJqIim4Lb1LqQA-eAMnYiNh1oaxUw238d5_lLOnwKd_MObNjm7_JRUpwcwYUZ0IilyhME33uTwgHjFe0lrdsEZxwKzbvyspjse36k5hqCX_tjwscrZq7TndBkaHJET4DsC5SPMV24ptNkJDLLTjX5TFZrqctXwbcbyxRfWFvegGDcHkoZkDvbYV-3gW0itod_5630LUgvPMZBKl4UK2EsdEB4CiaALNwVuFXV-e0vQXIDM8daIQ_PyUXxtNDs0nDE0VF8melfRpnkIjD-NpxbLbTmnPuhvlB9aa8wngrZuYy9bVvUDLkDiXspqlHTpGYCk5NTIYDveeY-GaIQhTJUTbLPghenpWL8qaYkHkokumWOBlOzapAJXMl5O5HwCB8Q8M_XjnJmtfHYj3SkMznOp4EfHCtj4796vIkW2kEKMvfak1MFD0OBqwmYIg12oMYa83QmoLyGYLTq_oc8B4E5amIZ3lk5zpKPXoBAq1B9UULSutlhKm1I78pytcM9wAJhuD5TdrS1RtcdzJP3s2-tBiPwhR5Ubk_BCuaQ30qEMLzKJX0i5E9KE6kHA8nkY6xQbeFUYFVm4ZOHMoZ_KdNmAA8dWKIHZgL3pfjJMQN4GSWhhq4NyRe2AFaw0afmiM5SIcilWhKpKAY_7TOHmBxHNcqw2dmaUALSyqBVuQvd8y368XPUiIDmOpIQ7juYdD5oXoUXT_8WRRgiIGJ1p8ZWS4l2LmSbiVTGwZ61N0-QxFgjIRGrTa9-7O6kEqiO79Zc7dTYJYJxcCXfbTuZ42N-pWsfdOizCOe0PJ-PV4bjsxAD6W7JC3yVt8EVINrreYC_B4CKh3e6XGbQ_pY76m1Eaa4wMPgVcx6gLU5syNnguJKqwS3vGP8Bxc3ZBifodzNw2QODv8H-e_g7DKrH5W1Y70Hcr5_hmjABi2vryEFUKng=w1920-h902)

### Dialogue Policies

The configuration of our dialogue policies inside the `config.yml` file:

```yaml
policies:
  - name: MemoizationPolicy
  - name: RulePolicy
  - name: TEDPolicy
    max_history: 5
    epochs: 100
```

- `MemoizationPolicy`
    
    This policy memorizes the conversations from the training data. It checks if the current conversation matches the stories defined in the `stories.yml` file, and predicts the next action from the matching stories with a confidence level of 1, otherwise, it predicts action `None` with confidence 0.
    
- `RulePolicy`
    
    This policy uses the rules defined in the `rules.yml` file to make fixed predictions upon encountering an utterance that is defined in the mentioned file.
    
- `TEDPolicy`
    
    This is the policy responsible for generalizing to unseen conversation paths.
    
    The input to the TED architecture is a features vector composed of:
    
    - The featurization of the previous message intent.
    - The featurization of the previous message entities.
    - The featurization of the bot’s slots.
        
        The featurization of a slot differs based on its type. For example, a slot of type `list`, like the `colleted_observations` slot, affects the final features vector depending on whether the list is empty or.
        
    - The featurization of the bot’s previous actions.
    
    The model has to learn a mapping from this input features vector to the correct action as defined in the `stories.yml` file.
    
    The featurization could be one-hot encoding or categorical encoding.
    

![Illustration explaining the nature of input and output for TED model](https://lh3.googleusercontent.com/fife/AAWUweV3DOdpoOTItTWl3zObplG-L_uayNyhurlUvJL8jv-L-gMIFo0iSrloqg3lL-T_kDV9fXjXu0JaGBj6h2j_RitUYQAqJf1E8mv3SAR9R6aC1Tf0nLIpKmdbvgHDN7US35HCB3x57IwmjngMy_Hor2Shk-O6FCSsV98x5s1Ua96-b_OLrKwNc6MY89xl3Cq89luPW7oArr1QsAdMHW4K1QdzsNtocOI7-RhLlY1IqMdqGph9ZBP9bvjuvtm63qYrkHv2oLbGzx2nguizkHDoT2udI6gbxCHFaERqM2XznI9NYhKNzf04NlztPOG5HZfObGAtUEG3Q2X3OOyZX2DIEyPDMSAfcwquGDukNE0CG8mASJ11tpdZvokKy0Yq0NWsNwZ-SP_0JL59broACEu7YLpKKk6BKF_0gOPGG83rScufQ1Tp3jiv7KcQAAdzUIA0geBw8Df82btzlgKO9OMCYL0EbjqWtnbH0GE_JwZdAgzWStFyxR49I0pEhUjtfYNrYLkySnqyJDDUeXEr8fpWU7RsZg5bdYN_7fGlHcQHvCH7gezZWdDmyqrueQPuoNR5-Tf5woROdoAebBLI2CTBt4WkiM6Wn3hGikchNnHAeVLhLJH6--mMX1Qop1XeW8SJJtl8HwMxEeHRyqXSNQcWrr9m7hxdaLqCMyJV5Y_-tEI9SIhdbCwfmDnefU39kGK18iBturPqRtXAJnjQL9rU5OoAk1Tbwi50MI5NKOKVFIrVIXtucEP06Bu7wp1X_DzejtLRyo74RMnvVl3PPfBOF_yPMuFQ0L_9VGqGc6uNmQg6QuPngLX9gY9JJdf3eg3pIKfQTsxdMv3dL-iShZJKh3cq1aOFWpWMiPAZHj6XEZok8jn0wzGUsJ0pCk4F0NilLI0qHQoBVQ2dNepy1kkLCHfiKnIKY4JP_Sy38StKPg7yLSbgP102WrNUE-9OQt_lgkmhvsMVenhfBmzoq0P1k3ZuzCSjkMVasuAkVsR6d_MMr7hkisLLKG3pjEqcdNfPC3KCEIBHVu_lBV7gHfKiD-DyyjGZuR3wV9l1I8RHBiIrSt_rmQVZFqUklexxZLAefLS6ZpJHVMm9k-0S0hpR4L3lCJYI0KLaqSFxmj5c7N_Wx6TEoVwFlk9lXBhXXaPmk4z7IZ_9FzY9vGKmHO8gwoVU6wVw6PNhGjIk_cRdZe3QsBL81-myAB0W1i7GjHAI7Udy66BYgVPK5I69TVh1uHcE09-MccaGekKjRPI4T-BA0D8cwKi4Oi5B6GWDcvWOuyWoJA5cQ6TulPBQZonb9cM4TUVf33Pp70a5k0ejDCA5_jHPyYbyJk9T6TL9eyXqG7ep4crd525_jMLNAM6UMZxvi7OracxVVXQoRZrTYBNYsbhXGVtkyw=w1920-h902)

Illustration explaining the nature of input and output for TED model

# Conclusion

We learned a lot about building a chatbot from this project. We can confidently say that we have gone from zero to hero in the task of building a conversational agent. We learned about many NLP concepts and all the phases important to building an AI project, starting from the idea until deployment. It’s true that we didn’t get the most optimal results for this task, but it was definitely a successful experience, and if we ever decided to pursue Hakim in the future, now we know where to start from, and what are the most important problems that need addressing.