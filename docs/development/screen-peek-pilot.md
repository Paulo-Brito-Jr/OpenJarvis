# Piloto "Screen Peek" — especificação

> Status: **especificação** (nenhum código neste PR). Origem: tarefa Skynet
> `tsk_4Z1IOnNjwh` (F3 — Olhos contínuos). Decisão do Paulo em 01/10/2026:
> **"Só tela, manual, sem retenção."**

Este piloto substitui, por ora, o escopo amplo da F3 (screen-watch contínuo,
câmera do Air, câmeras da casa). Tudo que não está listado em
[Escopo permitido](#escopo-permitido) está **fora** do piloto.

## Objetivo

Permitir que o Paulo peça, **de forma explícita e pontual**, que o OpenJarvis
olhe a tela dele **uma vez** e responda uma pergunta sobre o que viu
("o que está nesse erro?", "resuma essa janela"). Depois da resposta, nada do
que foi visto permanece.

## Escopo permitido

| Item | Regra |
|---|---|
| Acionamento | **Manual, um disparo por comando.** Ex.: `jarvis screen-peek "pergunta"`. Cada comando = uma captura = uma resposta. |
| Fonte | **Somente captura de tela** (display escolhido pelo usuário; padrão = monitor principal). |
| Frequência | Nenhuma. Sem intervalo, sem loop, sem "continue observando". |
| Processamento | Imagem lida e enviada ao modelo de visão **local** (Ollama em `127.0.0.1`), conforme a política de instalação segura do Air (engine somente local). |
| Retenção | **Zero.** Ver [Sem retenção](#sem-retencao). |

## Fora do escopo (proibido no piloto)

- **Câmera** (FaceTime ou qualquer outra) e **microfone/áudio**.
- Câmeras da casa (Home Assistant) e captura de outra máquina da frota (KVM).
- **Modo contínuo / watch / agendado / acionado por gatilho** (cron, evento,
  palavra de ativação, regex na pergunta etc.). Só o comando manual dispara.
- Qualquer envio da imagem para fora do equipamento (cloud, gateway remoto,
  provider externo). Se o modelo local não estiver disponível, o comando
  **falha fechado** com mensagem clara; não há fallback para nuvem.
- Persistir a imagem, miniatura, OCR/texto extraído, embedding ou descrição em
  disco, banco, memória de longo prazo (`memory`), traces, telemetria ou logs.
- Captura silenciosa de janelas de outros usuários/sessões; captura de telas
  de senha/cofre/gerenciador de credenciais (ver [Consentimento](#consentimento)).

## Sem retenção

<a id="sem-retencao"></a>

Requisitos verificáveis:

1. **Captura em memória.** A imagem nasce e vive em um buffer de processo
   (bytes). Não usar arquivo temporário (nem `/tmp`, nem `$TMPDIR`). Mecanismos
   aceitáveis: API nativa (Quartz `CGDisplayCreateImage`) ou `screencapture`
   escrevendo em *pipe* anônimo/`stdout`, nunca em caminho de arquivo.
2. **Redução em memória.** Redimensionar/comprimir (ex.: largura ≤ 1100 px,
   JPEG) no próprio buffer antes de enviar ao modelo.
3. **Descarte determinístico.** Ao fim do comando — inclusive em erro,
   timeout ou `Ctrl+C` — zerar/soltar o buffer (`try/finally`). Sem *cache*.
4. **Resposta efêmera.** A resposta vai para o terminal (stdout) e **não** é
   gravada em `memory`, traces, `audit.db` com conteúdo, nem histórico de chat.
   O registro permitido é só **metadado de auditoria** (ver abaixo).
5. **Sem swap intencional.** Não usar `mmap` para arquivo; não habilitar dump de
   buffer em modo debug.
6. **Metadado de auditoria permitido (sem conteúdo):** timestamp, display
   escolhido, tamanho em bytes do buffer, modelo usado, duração, resultado
   (`ok`/`erro`/`negado`). Nada de pergunta literal se ela puder conter dado
   sensível — registrar só o tamanho em caracteres.

### Como provar (critérios de aceite)

- Teste automatizado: executar o comando com *monkeypatch* na captura e
  afirmar que **nenhum arquivo novo** aparece em `$TMPDIR`, `/tmp`, no diretório
  de dados do OpenJarvis (`~/.openjarvis`) nem no diretório corrente.
- Teste de falha: erro do modelo, modelo ausente e `KeyboardInterrupt` também
  deixam zero arquivos e liberam o buffer.
- Teste de rede: com o modelo local, **nenhuma** conexão que não seja
  `127.0.0.1:11434`.
- Auditoria: `audit.db` e `traces.db` não contêm bytes de imagem nem texto da
  resposta após uma execução.

## Consentimento

<a id="consentimento"></a>

1. **Opt-in explícito e por comando.** O ato de digitar o comando é o
   consentimento daquela captura; não existe "lembrar permissão". Sem flag que
   pule a confirmação.
2. **Feature desligada por padrão.** Habilitar exige o usuário ligar
   `screen_peek.enabled = true` na configuração local (modo 600). Ausente =
   desligado = comando recusa.
3. **Aviso visível antes da captura.** Imprimir qual display será lido e que a
   imagem será analisada **localmente e descartada**; aguardar confirmação
   interativa (`y/N`) — não confirmável por agente/automação (exige TTY).
4. **Indicador durante o uso.** Mostrar "capturando…" e "descartado" ao fim;
   some o indicador quando termina.
5. **Quem pode consentir.** Apenas o dono da sessão local (Paulo). Familiares ou
   outras contas **não** são cobertos por este consentimento; a captura é do
   display do usuário que digitou o comando.
6. **Conteúdo sensível.** Antes de capturar, se a janela em foco for de
   gerenciador de senhas, cofre (`vault`), tela de login ou terminal com
   segredos conhecidos, o comando **recusa** (lista de bloqueio configurável).
7. **Revogação.** Basta `screen_peek.enabled = false`. Não há dado retido para
   apagar.
8. **Permissão do macOS.** Usa a permissão "Gravação de Tela" do TCC já
   concedida ao processo; o piloto **não** a solicita sozinho nem a automatiza
   por cliques. Se ausente, falha com instrução para o usuário conceder
   manualmente.

## Desenho proposto (para a implementação futura)

- Módulo novo isolado (`openjarvis.tools.screen_peek`), registrado **fora** da
  lista de ferramentas de agente: chamável só pelo comando de CLI, nunca por um
  agente/LLM via *tool call* (evita acionamento por injeção de prompt).
- Política: `ScreenPeekPolicy` fail-closed (desligado, sem rede externa, sem
  arquivo, sem agente).
- Modelo de visão local: um VLM pequeno servido pelo Ollama já instalado. A
  escolha do modelo segue o protocolo de supply chain da instalação (cooldown de
  7 dias, OSV, digest fixado).
- Plataforma: macOS primeiro (Air e Pro). Sem Windows/Linux no piloto.

## Rollback

Spec apenas: reverter este PR remove o documento. Quando houver código, o
rollback é `screen_peek.enabled = false` (efeito imediato) e *revert* do PR de
implementação; não há dados a migrar porque nada é retido.

## Relação com a Skynet

O `sky-percebe` do repositório Skynet hoje grava capturas em `/tmp`, aceita
`--camera` e `--watch` e usa visão em nuvem. **Ele não atende a este piloto** e
não deve ser usado como implementação: o piloto exige captura em memória, modo
local e disparo estritamente manual. A implementação em OpenJarvis é um
follow-up separado, com este documento como critério de aceite.

## Pendências para sair da especificação

- [ ] Paulo confirmar o display padrão e a lista de bloqueio de janelas.
- [ ] Escolher e auditar o modelo de visão local (supply chain).
- [ ] Implementação mínima + testes dos critérios de aceite acima.
