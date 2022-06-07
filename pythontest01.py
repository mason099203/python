import tkinter as tk
import time
import cv2
from PIL import Image, ImageTk
import csv
import copy

from PIL import Image, ImageTk
import numpy as np
import math
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = 'D:\Tesseract\\tesseract.exe'
import difflib as diff

set_check_time = 10  # 密碼確認次數

def creatinput(name, type):
    global set_check_time
    set_place = 150

    keyboard = tk.Tk()
    if type < 2:
        keyboard.title(name)
    else:
        name = '確認密碼剩餘次數 : ' + str(abs(type - set_check_time -1))
        keyboard.title(name)
    keyboard.geometry("790x450+0+0")
    keyboard.resizable(width=0, height=0)

    t1 = tk.Text(keyboard, width=13, height=1, font=('Arial', 30, 'bold'))
    t1.place(x=set_place + 150, y=12)

    l1 = tk.Label(keyboard, text=name, font=('Arial', 15, 'bold'))
    l1.place(x=set_place - 70, y=12)

    def quit_window():
        keyboard.destroy()

    quit = tk.Button(keyboard, text='離開', width=10, height=1, command=quit_window, font=('Arial', 10, 'bold'))
    quit.place(x=700, y=50)

    class Test(tk.Button):

        def __init__(self, t):
            super().__init__(keyboard)
            self.start, self.end = 0, 0
            self.set_down()
            self.set_up()
            self['text'] = chr(int(t))
            self['command'] = self._no_op
            self['font'] = ('Arial', 30, 'bold')
            self['width'] = 2
            self['height'] = 1

        def _no_op(self):
            """keep the tk.Button default pressed/not pressed behavior
            """
            pass

        def set_down(self):
            self.bind('<Button-1>', self.start_time)

        def set_up(self):
            self.bind('<ButtonRelease-1>', self.end_time)

        def start_time(self, e):
            self.start = time.time()

        def end_time(self, e):
            if self.start is not None:  # prevents a possible first click to take focus to generate an improbable time
                self.end = time.time()
                self.action()
            else:
                self.start = 0

        def action(self):
            var = self['text']
            t1.insert('insert', var)

            if type == -2:
                tocheck_password.append(self.start)
                tocheck_password.append(self.end)
            if type > 0 :
                check_Password[type - 1].append(self.start)
                check_Password[type - 1].append(self.end)

            self.start, self.end = 0, 0

    def del_button():
        t1.delete("1.0", "end")
        if type > 0:
            check_Password[type - 1] = []
    b_num = [0] * 10
    t, x, y = 48, 25, 70
    if type < 1 and type != -2:
        for i in range(10):
            b_num[i] = Test(t)
            b_num[i].place(x=x, y=y)
            t += 1
            x += 75
    else:
        b_num = [0] * 10
        t, x, y = 48, 25, 70
        for i in range(10):
            if i == 0:
                b_num[i] = Test(t)
                b_num[i].place(x=310 + 75, y=280)
                t += 1
            if i ==1 or i==2 or i==3:
                b_num[i] = Test(t)
                b_num[i].place(x=310+75*((i-1)%3), y=70)
                t += 1
            if i ==4 or i==5 or i==6:
                b_num[i] = Test(t)
                b_num[i].place(x=310+75*((i-1)%3), y=140)
                t += 1
            if i==7 or i==8 or i==9:
                b_num[i] = Test(t)
                b_num[i].place(x=310+75*((i-1)%3), y=210)
                t += 1

    beng = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'Z', 'X',
            'C', 'V', 'B', 'N', 'M']
    b_eng = [0] * 26
    if type < 1 and type != -2:

        t, x, y = 65, 25, 8
        for i in range(26):
            t = ord(beng[i])
            if i < 10:
                b_eng[i] = Test(t)
                b_eng[i].place(x=x, y=160)
            elif i < 19:
                b_eng[i] = Test(t)
                b_eng[i].place(x=x - 720, y=250)
            else:
                b_eng[i] = Test(t)
                b_eng[i].place(x=x - 1300, y=340)
            t += 1
            x += 75

    b_del = tk.Button(keyboard, text='清除', width=4, height=1, bg='red', command=del_button, font=('Arial', 14, 'bold'))
    b_del.place(x=set_place + 450, y=10)

    def getText():
        global password, account, check_time, register_account

        #登入帳號
        if type == -1:
            if t1.get("1.0", "end") == '\n':
                account = '空的'
            else:
                account = t1.get("1.0", "end")
                text_account.insert('insert', account)
            keyboard.destroy()
        if type == -2:
            if t1.get("1.0", "end") == '\n':
                password = '空的'
                check_time = '0'
            else:
                password = t1.get("1.0", "end")
                text_password.insert('insert', password)
                # time_end = time.time()
                # check_time = time_end - time_start
            keyboard.destroy()

        if type == 0:#帳號
            # global register_account
            register_account = t1.get("1.0", "end")
            print('register_account:' + register_account)
            keyboard.destroy()
            creatinput('註冊:密碼', type+1)

        if type > 0 and type < set_check_time:#密碼
            # global register_password
            register_password[type - 1] = t1.get("1.0", "end")
            if type > 1 and type <set_check_time:
                if register_password[type - 1] != register_password[type - 2]:
                    keyboard.destroy()
                    creatinput('註冊:確認密碼', type)
            print(register_password)
            print('type' + str(type))
            keyboard.destroy()
            creatinput('註冊:確認密碼', type+1)



        if type == set_check_time:
            # global register_password
            register_password[type - 1] = t1.get("1.0", "end")
            if register_password[type - 1] != register_password[type - 2]:
                keyboard.destroy()
                creatinput('註冊:確認密碼', type)
            print(register_password)
            keyboard.destroy()

            register = tk.Tk()
            register.title(register)
            print('type' + str(type))
            register.geometry("790x450+0+0")
            register.resizable(width=0, height=0)
            # 驗證帳號
            if True:
                # if register_password == register_password:
                password_len = len(register_password[0]) - 1
                l6 = tk.Label(register, text='註冊成功', font=('Arial', 10, 'bold'))
                l6.place(x=100, y=100)
                copy_check = copy.deepcopy(check_Password)
                for t in range(3):
                    copy_check[t].insert(0, 0)
                    copy_check[t].pop()

                for i in range(set_check_time):
                    for k in range(len(copy_check[0])):
                        # print(copy_check[i][k],check_Password[i][k])
                        check_Password[i][k] = check_Password[i][k] - copy_check[i][k]
                for t in range(set_check_time):
                    check_Password[t].pop(0)
                # print('check_Password',check_Password)
                mean_RP_1, mean_RP_2 = 0, 0
                mean_PR_1, mean_PR_2 = 0, 0
                var_RP_1, var_RP_2 = 0, 0
                var_PR_1, var_PR_2 = 0, 0
                for i in [0, 1]:  # mean
                    for k in range(len(check_Password[i])):
                        if i == 0:
                            if k % 2 == 0:
                                mean_RP_1 += check_Password[i][k]
                            else:
                                mean_PR_1 += check_Password[i][k]
                        if i == 1:
                            if k % 2 == 0:
                                mean_RP_2 += check_Password[i][k]
                            else:
                                mean_PR_2 += check_Password[i][k]
                mean_RP_1, mean_RP_2 = mean_RP_1 / (password_len - 1), mean_RP_2 / (password_len - 1)
                mean_PR_1, mean_PR_2 = mean_PR_1 / password_len, mean_PR_2 / password_len
                mean_PR = (mean_PR_1 + mean_PR_2) / 2
                mean_RP = (mean_RP_1 + mean_RP_2) / 2
                print('mean_PR_1, mean_PR_2,mean_RP_1,mean_RP_2', mean_PR_1, mean_PR_2, mean_RP_1, mean_RP_2)
                for i in [0, 1, 2]:  # var
                    for k in range(len(check_Password[i])):
                        if i == 0:
                            if k % 2 == 0:
                                var_RP_1 += abs(mean_RP_1 - check_Password[i][k])
                            else:
                                var_PR_1 += abs(mean_PR_1 - check_Password[i][k])
                        if i == 1:
                            if k % 2 == 0:
                                var_RP_2 += abs(mean_RP_2 - check_Password[i][k])
                            else:
                                var_PR_2 += abs(mean_PR_2 - check_Password[i][k])
                var_RP_1, var_RP_2 = var_RP_1 / (password_len - 2), var_RP_2 / (password_len - 2)
                var_PR_1, var_PR_2 = var_PR_1 / (password_len - 1), var_PR_2 / (password_len - 1)
                var_PR = (var_PR_1 + var_PR_2) / 2
                var_RP = (var_RP_1 + var_RP_2) / 2
                print('var_PR_1, var_PR_2,var_RP_1,mean_RP_2', var_PR_1, var_PR_2, var_RP_1, var_RP_2)

                with open('data.csv', 'a', newline='') as csvfile:
                    fieldnames = ['account', 'password', 'mean_RP', 'mean_PR', 'var_RP', 'var_PR']
                    writer = csv.DictWriter(csvfile, fieldnames)  # 建立 CSV 檔寫入器
                    # writer.writeheader()
                    writer.writerow({'account': register_account[:-1], 'password': register_password[0][:-1], 'mean_PR': mean_PR,
                                     'mean_RP': mean_RP, 'var_PR': var_PR, 'var_RP': var_RP})
                # with open('data.txt', 'a') as f:
                #     f.writelines(a)


            l4 = tk.Label(register, text='車牌' + register_account, font=('Arial', 10, 'bold'))
            l4.place(x=0, y=0)
            l5 = tk.Label(register, text='密碼' + register_password[0], font=('Arial', 10, 'bold'))
            l5.place(x=0, y=50)

            # with open('data.csv', 'a', newline='') as csvfile:
            #     fieldnames = ['C1','C2']
            #     writer = csv.DictWriter(csvfile,fieldnames)  # 建立 CSV 檔寫入器
            #     writer.writeheader()
            #     writer.writerow({'C1':check_Password[0],'C2':check_Password[1]})

            # print(check_Password[0] + check_Password[1])

            # print(check_Password[0])
            # print(check_Password[1])
            def quit():
                global check_Password
                check_Password = [[], [], []]
                register.destroy()

            register_quit = tk.Button(register, text='離開', width=10, height=1, command=quit, font=('Arial', 10, 'bold'))
            register_quit.place(x=700, y=50)
            register.mainloop()

    B_Enter = tk.Button(keyboard, text='確認', width=4, height=1, bg='green', command=getText, font=('Arial', 14, 'bold'))
    B_Enter.place(x=set_place + 530, y=10)

    # keyboard.bind('<Motion>',motion)
    keyboard.mainloop()
def quit():
    global check_Password
    check_Password = [[], [], []]
    root.destroy()
def register():
    global check_Password, register_password, set_check_time
    register_password = []
    check_Password = []
    for i in range(set_check_time):
        register_password.append([])
        check_Password.append([])
    creatinput('註冊:車牌號碼', 0)
def enter_account():
    text_account.delete("1.0", "end")
    creatinput('車牌號碼', -1)
def enter_password():
    text_password.delete("1.0", "end")
    global tocheck_password
    tocheck_password = []
    creatinput('設定密碼', -2)
def login():
    global tocheck_password
    root_account = text_account.get("1.0", "end")
    root_password = text_password.get("1.0", "end")
    account_in_data = 0
    password_in_data = 0
    with open('data.csv', newline='') as csvfile:
        rows = csv.DictReader(csvfile)  # 讀取 CSV 檔案內容
        for row in rows:  # 以迴圈輸出每一列
            if root_account[0:-2] == row['account']:
                account_in_data = 1
                print('帳號正確')
                pr = []
                rp = []
                for i in range(1, len(tocheck_password), 2):
                    pr.append(tocheck_password[i] - tocheck_password[i - 1])
                for i in range(2, len(tocheck_password), 2):
                    rp.append(tocheck_password[i] - tocheck_password[i - 1])
                if root_password[0:-2] == row['password']:
                    password_in_data = 1
                    di = 0
                    dij = 0
                    for k in range(len(pr)):
                        di += abs(pr[k] - float(row['mean_PR']))/float(row['var_PR'])
                        dij += abs(pr[k] - float(row['mean_RP']))/float(row['var_RP'])
                        # print(di,dij)
                    #     設定T
                    t = (dij + di)/(len(root_password[0:-2])*2 - 1)




                    print('是否為本人',t)
                    break
                else:
                    print('密碼錯誤')
                break

    # if account_in_data == 0:
    #     print('帳號或密碼錯誤')
    # elif password_in_data == 0:
    #     print('密碼錯誤')
    # else:
    #     print('登入成功')

    print('車牌:' + root_account[0:-2] + '\n' + '密碼:' + root_password[0:-2])


root = tk.Tk()
root.title("Tk GUI")
root.geometry('800x480')

text_password = tk.Text(root, width=13, height=1, font=('Arial', 18, 'bold'))
text_password.place(x=200, y=50)
text_account = tk.Text(root, width=13, height=1, font=('Arial', 18, 'bold'))
text_account.place(x=200, y=10)

button_enter1 = tk.Button(root, text='輸入 車牌號碼', width=15, height=1, command=enter_account, font=('Arial', 11, 'bold'))
button_enter1.place(x=40, y=10)
button_enter2 = tk.Button(root, text='輸入 密碼', width=10, height=1, command=enter_password, font=('Arial', 11, 'bold'))
button_enter2.place(x=40, y=50)

button_enter3 = tk.Button(root, text='測試帳號密碼', width=10, height=1, command=login, font=('Arial', 10, 'bold'))
button_enter3.place(x=500, y=10)
button_quit = tk.Button(root, text='離開', width=10, height=1, command=quit, font=('Arial', 10, 'bold'))
button_quit.place(x=700, y=10)
button_register = tk.Button(root, text='註冊', width=10, height=1, command=register, font=('Arial', 10, 'bold'))
button_register.place(x=700, y=50)


class Mainapplication_A(tk.Frame):  # txt+滾輪
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.pack()
        self.createwidget()
        self.openfile()

    def createwidget(self):
        self.scrollbar = tk.Scrollbar(self)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.text = tk.Text(self, width=30, height=10, yscrollcommand=self.scrollbar.set)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH)
        self.scrollbar.config(command=self.text.yview)

    def openfile(self):
        # with open('data.txt', 'r') as f:
        #     temp = f.read()
        with open('data.csv', newline='') as csvfile:
            rows = csv.DictReader(csvfile)  # 讀取 CSV 檔案內容
            self.text.insert(tk.END, 'account   password'+'\n')
            for row in rows:  # 以迴圈輸出每一列
                # print(row['account'])
                self.text.insert(tk.END,row['account']+'    '+row['password']+'\n')


text_A = Mainapplication_A(master=root)
text_A.place(x=550, y=150)

video_canvas = tk.Canvas(root, width=400, height=300, bg='#CDB38B')
video_canvas.place(x=20, y=120)
set_v = 0
def set_video():
    global set_v,cap
    if set_v == 0:
        set_v = 1
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        # cap = cv2.VideoCapture('test0511.h264')
        video_imd()
    else:
        set_v = 0
        cap.release()

button_video = tk.Button(root, text='video', width=10, height=1, command=set_video, font=('Arial', 10, 'bold'))
button_video.place(x=40, y=90)

# -------影片
# video_canvas = tk.Canvas(root, width=700, height=400, bg = '#CDB38B')
# video_canvas.place(x=20, y=100)
def rota_img(img):
    ret2, binary_dst_inv = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    coords = np.column_stack(np.where(binary_dst_inv > 0))
    angle = cv2.minAreaRect(coords)[2]
    d_h, d_w = img.shape
    if angle > 45:
        angle = 90 - angle
    else:
        angle = -angle
    center = (d_w // 2, d_h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (d_w, d_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return  rotated

def carnum2(img_o):
    global text,time_now
    # img_o = img_o[600:1100,800:1100]
    # img_o = img_o[100:400,100:400]
    new_img = img_o.copy()
    # print(new_img.shape)
    gray_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)
    binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 11, 2)
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    pi = []
    for c in contours:
        x = []
        y = []
        for point in c:
            y.append(point[0][0])
            x.append(point[0][1])
        r = [min(y), min(x), max(y), max(x), (max(x) + min(x)) // 2, (max(y) + min(y)) // 2]

        if cv2.contourArea(c) < 200 or cv2.contourArea(c) > 4*300 or float((r[3]-r[1])/(r[2]-r[0])) < 0.5 or float((r[3]-r[1])/(r[2]-r[0])) > 7.5:
            contours = np.delete(contours,count,0)
            count = count - 1
        else:
            pi.append(r)
        count = count + 1

    # 畫刪除後的輪廓圖
    # contours_cut_img = new_img.copy()
    contours_cut_img = cv2.drawContours(new_img.copy(), contours, -1, (0, 255, 0), 2)

    # 區塊分類
    count_p = [[0]]
    for i in range(1, len(contours)):
        for j in range(len(count_p)):
            i_h = pi[i][3] - pi[i][1]  # 首塊高度
            j_h = pi[count_p[j][0]][3] - pi[count_p[j][0]][1]  # 檢視的高度
            # 首塊中心 pi[i][4],pi[i][5]
            ij_l = math.sqrt((pi[i][4] - pi[count_p[j][0]][4]) ** 2 + (pi[i][5] - pi[count_p[j][0]][5]) ** 2)
            if pi[i][5] - pi[count_p[j][0]][5] == 0:
                ij_m = 999
            else:
                ij_m = (pi[i][4] - pi[count_p[j][0]][4]) / (pi[i][5] - pi[count_p[j][0]][5])

            if i_h / j_h > 0.8 and i_h / j_h < 1.2 and ij_l < 4*i_h and ij_m < 0.3 and ij_m > -0.3:
                count_p[j].append(i)
                break
            if j == (len(count_p) - 1):
                count_p.append([i])
    # print(count_p)

    for i in range(len(count_p)):
        # 取出區塊中6-7
        count_img_plate = 0

        if len(count_p[i]) < 8 and len(count_p[i]) > 5:
            # 依照y大小牌順序
            for ii in range(len(count_p[i])):
                for jj in range(len(count_p[i])):
                    y_ii = []
                    for point in contours[count_p[i][ii]]:
                        y_ii.append(point[0][0])
                    y_ii = min(y_ii)
                    y_jj = []
                    for point in contours[count_p[i][jj]]:
                        y_jj.append(point[0][0])
                    y_jj = min(y_jj)
                    if y_jj > y_ii:
                        t = count_p[i][jj]
                        count_p[i][jj] = count_p[i][ii]
                        count_p[i][ii] = t
            x_all = []
            y_all = []

            for j in range(len(count_p[i])):

                x = []
                y = []
                for point in contours[count_p[i][j]]:
                    y.append(point[0][0])
                    x.append(point[0][1])
                r = [min(y), min(x), max(y), max(x)]
                x_all.append(r[1])
                x_all.append(r[3])
                y_all.append(r[0])
                y_all.append(r[2])
            # print("count_p",count_p[i])

            r_all = [min(y_all), min(x_all), max(y_all), max(x_all)]

            out_img = new_img.copy()

            mask = np.zeros(out_img.shape[:2], np.uint8)
            for k in range(len(count_p[i])):
                cv2.drawContours(mask, contours, count_p[i][k], 255, -1)
            dst = cv2.bitwise_and(out_img, out_img, mask=mask) + 255
            # print('dst',type(dst))
            if r_all[1] > 10 and r_all[0] > 10 and r_all[3]+10 < binary_img.shape[0] and r_all[2]+10 < binary_img.shape[1]:
                dst = dst[r_all[1]-10:r_all[3]+10,r_all[0]-10:r_all[2]+10]
            else:
                dst = dst[r_all[1]:r_all[3],r_all[0]:r_all[2]]
            gray_dst = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)

            ret2, binary_dst = cv2.threshold(gray_dst, 150, 255, cv2.THRESH_BINARY)
            binary_dst = rota_img(binary_dst)
            # cv2.imshow("binary_dst", binary_dst)
            # image(binary_dst)
            count_img_plate += 1
            text = pytesseract.image_to_string(binary_dst)
            re_text = ''
            for p in text:
                if p == 'O':
                    re_text += '0'
                elif p == '|':
                    re_text += '1'
                elif p == 'I':
                    re_text += '1'
                elif p == ' ':
                    re_text += '-'
                else:
                    re_text += p
            print(re_text[0:-2])
            # with open("data_in_park.txt",'a') as f:
            #     f.write(re_text[0:-2]+'\n')
            # check(re_text[0:-2]) #確認



            cv2.rectangle(new_img, (min(y_all)-3, min(x_all)-3), (max(y_all)+3, max(x_all)+0), (0, 0, 255), 4)
            cv2.rectangle(contours_cut_img, (min(y_all), min(x_all)), (max(y_all), max(x_all)), (0, 0, 255), 3)
    # time_now = time.ctime()

    # cv2.putText(contours_cut_img, time_now, (0,15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    # cv2.waitKey(0)
    return new_img,contours_cut_img


def video_imd():
    success, img = cap.read()
    n, c = carnum2(img)
    img_show = img.copy()
    # img_show[600:1100, 800:1100] = n
    # 640:480
    img = cv2.resize(img, (365, 274), interpolation=cv2.INTER_AREA)
    video_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 把图像从BGR转到RGB
    img_frame = ImageTk.PhotoImage(Image.fromarray(video_frame))
    video_canvas.create_image(200,150,image=img_frame)
    video_canvas.img = img_frame
    if set_v == 1:
        video_canvas.after(10, video_imd)  # 每30毫秒重置副程式
    else:
        cap.release()
# -----------
def update():  # 更新介面
    text = Mainapplication_A(master=root)
    text.place(x=550, y=150)
    root.after(1000, update)


root.after(1000, update)

root.mainloop()

#
#
# creatinput('帳號:車牌號碼',0)
# creatinput('密碼:password',1)
# print('[車牌,密碼]',[account,password])
# print('點擊位置',check_Password)
# print('點擊時間',check_time)
