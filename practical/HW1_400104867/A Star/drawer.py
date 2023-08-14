import matplotlib.pyplot as plt
import matplotlib.animation as pltani


def draw_elevator(ind, *args):
    ax, process, elevator = args
    first_stud, first_prof, second_stud, second_prof, e = process[ind]

    ax.collections.clear()

    ax.scatter(range(1, 1+first_stud), [1.5]*first_stud, color='blue', label='Students')
    ax.scatter(range(1, 1+first_prof), [2]*first_prof, color='green', label='Professors')

    ax.scatter(range(1, 1+second_stud), [5.5]*second_stud, color='blue')
    ax.scatter(range(1, 1+second_prof), [6]*second_prof, color='green')

    elevator.set_ydata([1.5, 1.5, 1.5] if e == 1 else [3.5, 3.5, 3.5])


def create_gif(solution):
    fig, ax = plt.subplots(figsize=(9, 5))

    # viz banks
    first_floor = plt.Rectangle((0, 0), 10, 1, color='brown', label='First Floor')
    second_floor = plt.Rectangle((0, 4), 10, 1, color='orange', label='Second Floor')
    plt.gca().add_patch(first_floor)
    plt.gca().add_patch(second_floor)

    ax.scatter([1, 2, 3], [2.5, 2.5, 2.5], color='blue', label='Students')
    ax.scatter([1, 2, 3], [3, 3, 3], color='green', label='Professers')
    ax.scatter([], [], color='blue')
    ax.scatter([], [], color='green')
    elevator = ax.plot([10, 11, 12], [1.5, 1.5, 1.5], linewidth=10, color='grey', label='Elevator')[0]

    ax.legend(loc='upper right')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)

    anim = pltani.FuncAnimation(fig, draw_elevator,
                                frames=len(solution),
                                interval=900,
                                fargs=(ax, solution, elevator))
    anim.save('result.gif')
