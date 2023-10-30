"""
    This is a program for user explanation to test improved interpretability of
    NewLIME.
"""

import tkinter
import tkinter.messagebox

import time

FILE_NAME = ''
question_num: int = 3


def title() -> None:
    """docstring"""
    ans = tkinter.messagebox.askokcancel(
            title='User Experiment',
            message='If you are ready, push the OK button.')
    if ans:
        info()


def info() -> None:
    """docstring"""
    global FILE_NAME

    def get_name() -> None:
        global FILE_NAME
        ans: bool = tkinter.messagebox.askokcancel(
                message=f'Your name is {entry.get()}. Is it OK?')
        if ans:
            t: int = int(time.time())
            FILE_NAME = f'{t}_result.txt'
            with open(FILE_NAME, 'a', encoding='UTF-8') as f:
                f.write('Name: ' + entry.get() + '\n')
            root.destroy()
            explanation()

    root = tkinter.Tk()
    root.title('Your information')
    root.eval('tk::PlaceWindow . center')

    label = tkinter.Label(text='Enter your name.')
    label.pack()

    entry = tkinter.Entry(width=20)
    entry.pack()

    button = tkinter.Button(text='OK', command=get_name)
    button.pack()

    root.mainloop()


def explanation() -> None:
    """docstring"""

    def call_question():
        root.destroy()
        question()

    root = tkinter.Tk()
    root.title('Generated Explanation')
    root.eval('tk::PlaceWindow . center')

    label = tkinter.Label(
        text='See the instance and the explanation for it. '
             'You can see them again after pushing OK button.')
    label.pack()

    button = tkinter.Button(
            text='OK',
            command=call_question)
    button.pack()

    root.mainloop()


def question(index: int = 1) -> None:
    """docstring"""

    def get_answer() -> None:
        ans = tkinter.messagebox.askokcancel(
                message=f'You selected {txt[answer.get()]}. Is it OK?')
        if ans:
            with open(FILE_NAME, 'a', encoding='UTF-8') as f:
                f.write(str(answer.get()) + '\n')
            root.destroy()
            if index < question_num:
                question(index + 1)
            else:
                tkinter.messagebox.showinfo(
                        title='Thank you!',
                        message='That\'s all.\n'
                                f'Please send me the file \'{FILE_NAME}\' '
                                'in your current directory.\n'
                                'Thank you for your help!\n\n'
                                'Genji Ohara')
    ##

    root = tkinter.Tk()
    root.title(f'Question {index}')
    root.eval('tk::PlaceWindow . center')

    label = tkinter.Label(
            text=f'Q{index}. What do you think the model\'s prediction to '
            'this instance?')
    label.pack()

    txt = ['Zero', 'One', 'I do not know']

    radio = []
    answer = tkinter.IntVar()

    for i, t in enumerate(txt):
        r = tkinter.Radiobutton(text=t, value=i, variable=answer)
        radio.append(r)
        r.pack()

    button = tkinter.Button(text='OK', command=get_answer)
    button.pack()
    root.mainloop()
##


title()
