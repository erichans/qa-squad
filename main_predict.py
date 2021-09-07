from transformers import BertForQuestionAnswering, BertTokenizerFast, pipeline


# dataset SQuAD v1 pt_BR: https://drive.google.com/file/d/1Q0IaIlv2h2BC468MwUFmUST0EyN7gNkn/view

def run():

    # model_type = 'base'
    model_type = 'large'

    tokenizer = BertTokenizerFast.from_pretrained(f'neuralmind/bert-{model_type}-portuguese-cased', local_files_only=True)

    # model = BertForQuestionAnswering.from_pretrained(
    #     f'C:/projetos/question-answering-squad-pt-br/results/2_epochs_{model_type}_qa/best_model_checkpoint-21940')

    model = BertForQuestionAnswering.from_pretrained(
        f'C:/projetos/question-answering-squad-pt-br/results/2_epochs_{model_type}_qa/best_model_checkpoint-10970')

    question_answerer = pipeline("question-answering", model=model, tokenizer=tokenizer)

    context = r"""
        O advogado Marcos Rogério Manteiga foi condenado a pagar indenização por danos morais no valor de R$ 10 mil 
    para a ex-comandante da Polícia Militar em Marília, Márcia Cristina Cristal Gomes.
        Na decisão, a Justiça ainda determinou que ele retire os vídeos publicados no Youtube e Facebook, no prazo de 
    cinco dias, contados do trânsito em julgado, sob pena de multa diária de R$ 500.
        Manteiga é advogado de defesa do sargento Alan Fabrício Ferreira, no caso da carteirada que ocorreu em agosto 
    do ano passado, quando a vereadora Professora Daniela ligou para a comandante após sua filha ter o veículo 
    apreendido em uma blitz na tentativa de evitar a apreensão. 
        Márcia Cristal, que na época comandava o policiamento do 9º Batalhão de Polícia Militar do interior da cidade, 
    ligou para o sargento e deu ordens para que o carro não fosse apreendido. O caso ganhou grande repercussão após o 
    advogado divulgar os áudios da conversa em que a comandante repreendia o sargento. 
        De acordo com a decisão da Justiça, "ainda que a autora (Márcia Cristal) tenha praticado conduta incompatível 
    em detrimento de cliente do requerido, tudo a ser apurado junto aos órgãos e instâncias competentes, tal fato não 
    pode ser usado, ainda mais em redes sociais com o acesso de várias pessoas ao conteúdo, como argumento para 
    declarações desrespeitosas e ofensivas, o que de fato foi feito pela parte requerida, o qual, repita-se, nem 
    mesmo alega que os vídeos e publicações indicados na peça inaugural por ele não foi publicado nas respectivas 
    plataformas".
        Isto é, apesar da conduta da ex-comandante, a sua exposição nas redes sociais gera declarações desrespeitosas e 
    ofensivas devido ao extenso número de pessoas alcançado pelas redes sociais.
        A TV TEM entrou em contato com Manteiga e ele afirmou que vai recorrer da decisão. O advogado ainda não retirou 
    os vídeos das plataformas, pois o caso ainda não transitou em julgado.
    """

    questions = ["Qual o nome do advogado?",
                 "Qual a condenação?",
                 "Qual o valor da condenação?",
                 "A quem será paga a indenização?",
                 "Qual o prazo para tirar o conteúdo ofensivo das redes sociais?",
                 "Qual o valor da multa se não tirar os vídeos?",
                 "Quem estava a frente do Batalhão de Polícia Militar?",
                 "Qual ordem foi dada pelo comandante do batalhão?",
                 "Por que os vídeos ainda não foram retirados das redes sociais?",
                 "Ainda cabe recurso da decisão?"]

    for question in questions:
        print(f"Pergunta: '{question}'")
        result = question_answerer(question=question, context=context)
        print(
            f"Resposta: '{result['answer']}'\n\tscore: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")




if __name__ == '__main__':
    run()

