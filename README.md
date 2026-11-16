
<h1 align="center">Resolução do Teste Técnico de IA para o processo seletivo da RabbITech</h1>

* <h2>Fluxograma do processo de decisão da IA e Funcionamento do RAG</h2>

<img width="2070" height="1106" alt="FluxogramaIA" src="https://github.com/user-attachments/assets/5722346a-cf49-477e-b0bf-b5bdd4d99bf4" />


## Como definir a confiança na Resposta da IA?
* [Similaridade de Cosseno do Banco Vetorial](https://www.ibm.com/br-pt/think/topics/cosine-similarity)
* Exemplo Visual:
<img width="488" height="376" alt="image" src="https://github.com/user-attachments/assets/ce227670-91f2-42fd-918a-cb3a8885caa8" />

* <h2>Pseudocódigo do Pipeline</h2>

O pseudocódigo abaixo segue o mesmo fluxo do fluxograma: **mensagem -> classificação de intenção -> busca por similaridade -> resposta ou escalonamento**.

```
função atender(mensagem):

    msg = mascarar_dados_sensiveis(mensagem)          # CPF, e-mail, telefone antes de qualquer log

    # 1. Classificação de intenção (saída em JSON, temperatura 0)
    intencao = classificar_intencao(msg)              # {categoria, pede_humano, urgencia}

    se intencao.pede_humano  OU  intencao.categoria == "fora_do_escopo":
        retornar escalar(msg, motivo = "intencao")

    # 2. Busca por similaridade (RAG)
    emb      = embed("search_query: " + msg)          # nomic-embed-text via Ollama (local)
    top_k    = buscar_indice(emb, k = 3)              # lista de (faq_id, similaridade_cosseno)
    melhor   = top_k[0]

    # 3. Decisão de confiança (retrieval)
    se melhor.similaridade < LIMIAR:
        retornar escalar(msg, motivo = "baixa_similaridade")

    # 4. Geração restrita ao contexto recuperado
    contexto = trechos_das_faqs(top_k acima do LIMIAR)
    saida    = llm_local(prompt_restrito(msg, contexto), temperatura = 0)   # {"resposta": texto | null}

    # 5. Decisão de confiança (geração)
    se saida.resposta == null:                        # modelo declarou não ter base para responder
        retornar escalar(msg, motivo = "sem_base_no_contexto")

    registrar(msg_mascarada, faq_id, similaridade, caminho = "automatico")
    retornar resposta_automatica(saida.resposta, faq_id)


função escalar(msg, motivo):
    criar_ticket_humano(msg, motivo, sugestao_de_faq = top_k)   # humano já recebe o contexto
    registrar(msg_mascarada, motivo, caminho = "escalonado")
    retornar "Vou encaminhar seu atendimento para um especialista."
```

**Os dois caminhos:**
* **Resposta automática** -> intenção ok + similaridade >= limiar + o modelo conseguiu responder usando *somente* o trecho da FAQ.
* **Escalonamento** -> qualquer uma das quatro saídas: cliente pediu humano, fora do escopo, baixa similaridade ou modelo sem base no contexto.

---

* <h2>Justificativa Técnica</h2>

### Por que Ollama (modelo local)?
- A **política de dados** da empresa proíbe enviar dados do cliente para APIs externas. Com o Ollama o modelo e as requisições ficam dentro do ambiente da RabbITech.
- O Ollama expõe uma API HTTP local, o que facilita integrar com o sistema de atendimento sem depender de terceiros.
- **Trade-off assumido:** um modelo 8B local (ex.: llama3.1 8B) é menos capaz que modelos grandes de API. Por isso o modelo **não responde "de cabeça"**: ele só reformula o que está na FAQ (ver abaixo).

### Por que RAG e não só o modelo?
- Um modelo sozinho responderia com o que aprendeu no treino, e não com o que a empresa realmente faz. Isso gera **alucinação** com tom de confiança.
- Com RAG a resposta nasce de um trecho do `faq_interno.md`, o que a torna **auditável** (sabemos qual FAQ originou a resposta) e **atualizável** (basta editar a FAQ e reindexar, sem retreinar nada).
- O RAG também nos dá um **sinal de confiança mensurável** (similaridade), que é a base da regra de negócio de escalonamento.

### Estratégia de embeddings e chunking
- **Modelo de embedding:** `nomic-embed-text`, rodando no próprio Ollama (continua 100% local). Serve para transformar o texto do documento faq_interno em vetores usando a similaridade de cosseno.
- **Chunking:** **1 pergunta + resposta = 1 chunk.** A FAQ já vem segmentada em unidades semânticas completas, então quebrar mais picotaria a resposta e misturar FAQs diferentes só diluiria a similaridade.
- **O que é embedado:** a pergunta (e, se os testes mostrarem ganho, pergunta + resposta). O cliente escreve algo parecido com a *pergunta*, então é ela que precisa casar.
- **Armazenamento:** com 18 FAQs, uma busca por força bruta em memória (cosseno com numpy) ou um SQLite/Chroma local basta. Um banco vetorial distribuído seria complexidade sem ganho; só passa a fazer sentido se a base crescer bastante.
- **Métrica:** similaridade de cosseno (já citada acima).

### Como medir e definir o limiar de confiança


**Pontos de atenção:**
- A similaridade **mede a busca, não a resposta**: a busca pode achar a FAQ certa e o modelo ainda assim gerar algo infiel. Por isso há uma **segunda barreira** (etapa 5 do pseudocódigo): o modelo é instruído a retornar `null` quando o trecho não contém a resposta.
- **Recalibrar sempre que trocar o modelo de embedding**, pois a escala das similaridades muda de um modelo para outro.
- Com apenas 18 FAQs o conjunto de validação é pequeno. O limiar inicial deve ser **conservador** e ajustado depois com as mensagens reais que foram escaladas e revisadas por humanos.

---

* <h2>Riscos Técnicos e Mitigações</h2>

| Risco | Por que importa neste cenário | Mitigação proposta |
|---|---|---|
| **Alucinação** | Um 8B local pode inventar passos, prazos ou valores que não estão na FAQ. | Prompt restrito ao contexto recuperado; temperatura 0; saída em JSON com `resposta: null` quando não houver base; limiar de similaridade calibrado; revisão amostral das respostas automáticas; mostrar ao cliente de qual FAQ veio a resposta. |
| **Latência do modelo local** | Sem GPU adequada, a geração pode ser lenta; o cliente fica esperando no atendimento inicial. | Manter o modelo carregado na memória (`keep_alive`); usar versão quantizada; contexto curto (só o top-k); **usar a similaridade como classificador barato** sempre que possível, evitando chamadas extras ao LLM; *streaming* da resposta; *timeout* -> **escalona** em vez de deixar o cliente sem retorno; cache para perguntas repetidas. Medir a latência real no hardware da empresa antes de prometer um SLA. |
| **Tamanho de contexto** | O Ollama tem um `num_ctx` padrão pequeno e, se o prompt exceder, **trunca silenciosamente**, podendo cortar justamente a instrução do sistema. | Definir `num_ctx` explicitamente (conferindo o valor padrão da versão instalada); limitar k e o histórico da conversa; medir o tamanho do prompt montado e alertar/escalar quando se aproximar do limite. |
| **Vazamento de dados** | A restrição principal do projeto é nada sair do ambiente. | Ollama escutando apenas em `localhost`/rede interna; sem APIs externas nem telemetria; mascarar dados sensíveis antes de logar; controle de acesso aos logs; **teste de rede** bloqueando saída para a internet e verificando que o pipeline continua funcionando. |
| **Classificação de intenção errada** | Uma reclamação grave tratada como dúvida de FAQ gera má experiência. | Pedido explícito de humano sempre escala; em dúvida, escalar; validar o classificador no mesmo conjunto de validação. |
| **Prompt injection** | O cliente pode escrever "ignore as instruções anteriores...". | Separar claramente instrução e mensagem do cliente no prompt; saída restrita ao formato JSON; resposta só é entregue se vier de um trecho recuperado. |
| **FAQ desatualizada** | A resposta correta hoje pode estar errada amanhã. | Processo de reindexação ao editar o `faq_interno.md`; versionar a FAQ junto com o índice. |

---

* <h2>Estimativa de Esforço</h2>

Estimativa para **1 desenvolvedor**, implementando o pipeline real. Os valores são estimativas minhas, apoiadas nas etapas abaixo; devem ser revistas após conhecer o hardware e o sistema de atendimento da empresa.

| Etapa | Dias | Justificativa |
|---|---|---|
| Ambiente: Ollama, modelos, hardware, rede local | 1 | Instalação, download dos modelos, configuração de `num_ctx`/`keep_alive` e isolamento de rede. |
| Ingestão e indexação da FAQ | 1 | Parser do `faq_interno.md`, chunking por pergunta/resposta, geração e armazenamento dos embeddings. |
| Classificação de intenção | 1,5 | Definir categorias, escrever o prompt com saída em JSON e testar. |
| Busca + calibração do limiar | 2 | Criar o conjunto de validação (positivos e negativos) e analisar as distribuições; é a etapa que sustenta a regra de negócio. |
| Geração com prompt restrito | 1,5 | Prompt anti-alucinação, tratamento do `null`, testes de fidelidade. |
| Escalonamento + integração com o atendimento | 2 | Criação do ticket com contexto para o humano; depende do sistema que a empresa usa (**premissa a confirmar**). |
| Logs, mascaramento de dados, observabilidade | 1 | Registro de decisões e métricas sem expor dados do cliente. |
| Testes e avaliação ponta a ponta | 2 | Medir precisão, taxa de escalonamento e latência; testes de injeção e de limite de contexto. |
| Deploy e documentação | 1 | Empacotamento, instruções de operação e reindexação. |
| **Subtotal** | **13** | |
| Contingência (hardware inesperado, ajustes de prompt) | 2 | Parte do risco está em medir o desempenho do modelo local, que só se confirma na prática. |
| **Total** | **15 dias** | |

---

* <h2>Uso de IA</h2>

- **Ferramentas usadas:** Claude (Anthropic) e Gemini(Google).
- **Em quais partes:** organização e redação do rascunho da justificativa técnica e do pseudocódigo com claude, também utilizei o gemini para esclarecer as diferentes alternativas de resolução para o problema e conhecer melhor sobre os problemas e dificuldades.

