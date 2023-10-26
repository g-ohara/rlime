# -*- coding: utf-8 -*-

import tkinter
import tkinter.messagebox

import time

question_num = 3

def title():
    ans = tkinter.messagebox.askokcancel(
            title='User Experiment',
            message=u'If you are ready, push the OK button.')
    if ans:
        info()

def info():

    def get_name() -> None:
        global file_name
        ans = tkinter.messagebox.askokcancel(
                message='Your name is %s. Is it OK?' % entry.get())
        if ans:
            t = int(time.time())
            file_name = '%d_result.txt' % t
            with open(file_name, 'a', encoding='UTF-8') as f:
                f.write('Name: ' + entry.get() + '\n')
            root.destroy()
            explanation()
    ##

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


def explanation():

    def call_question():
        root.destroy()
        question()

    root = tkinter.Tk()
    root.title('Generated Explanation')
    root.eval('tk::PlaceWindow . center')
    
    label = tkinter.Label(text=\
            'See the instance and the explanation for it. '\
            'You can see them again after pushing OK button.')
    label.pack()

    button = tkinter.Button(
            text='OK', 
            command=call_question)
    button.pack()
    
    root.mainloop()


def question(index: int = 1):

    def get_answer() -> None:
        global file_name
        ans = tkinter.messagebox.askokcancel(
                message='You selected %s. Is it OK?' % txt[answer.get()])
        if ans:
            with open(file_name, 'a', encoding='UTF-8') as f:
                f.write(str(answer.get()) + '\n')
            root.destroy()
            if index < question_num:
                question(index + 1)
            else:
                tkinter.messagebox.showinfo(
                        title='Thank you!',
                        message='That\'s all.\n'\
                                'Please send me the file \'%s\' '\
                                'in your current directory.\n'\
                                'Thank you for your help!\n\n'\
                                'Genji Ohara' % file_name)
    ##

    root = tkinter.Tk()
    root.title('Question %d' % index)
    root.eval('tk::PlaceWindow . center')
    
    label = tkinter.Label(
            text='Q%d. What do you think the model\'s prediction to this instance?' % index)
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
