
import matplotlib.pyplot as plt

def draw_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')  

    steps = [
        ("Início", (0.5, 0.9)),
        ("Aquisição dos Dados", (0.5, 0.75)),
        ("Preparação dos Dados", (0.5, 0.6)),
        ("Divisão Treino/Teste", (0.5, 0.45)),
        ("Treinamento do Modelo", (0.5, 0.3)),
        ("Avaliação do Modelo", (0.5, 0.15)),
        ("Aplicação em Produção", (0.5, 0.0)),
        ("Monitoramento e Retreinamento", (0.5, -0.15)),
    ]


    for step, (x, y) in steps:
        ax.text(x, y, step, fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", ec="navy", lw=2))


    for i in range(len(steps) - 1):
        start = steps[i][1]
        end = steps[i+1][1]
        ax.annotate("",
                    xy=(end[0], end[1] + 0.04),
                    xytext=(start[0], start[1] - 0.04),
                    arrowprops=dict(arrowstyle="->", lw=2, color='navy'))

    plt.title("Fluxograma do Pipeline do Projeto", fontsize=16, weight='bold', ha='center')
    plt.tight_layout()
    plt.savefig("pipeline_diagram.png")
    plt.show()

if __name__ == "__main__":
    draw_pipeline_diagram()
