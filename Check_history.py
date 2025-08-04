import database
import time

def del_channels(*nums):
    channels = database.get_all_channels()
    for i in nums:
        i -= 1
        channel_id = channels[i][0]
        database.delete_channel(channel_id)

def history(name):
    channels = database.get_all_channels()
    if 0 <= name < len(channels):
        channel_id = channels[name][0]
        messages = database.get_messages(channel_id)
        for msg in messages:
            print(f"{msg['role']}:\n")
            content = msg.get("content")
            if isinstance(content, list):
                for con in content:
                    if "text" in con.keys():
                        print(con["text"] + "\n")
                    elif "image_url" in con.keys():
                        print(f"{con["image_url"]["url"]}\n")
            elif isinstance(content, str):
                print(f"{content}\n")
    else:
        print("Некорректный номер канала.")

def opt_del():
    print("Введите номер каналы на удаление\n(введи 0 для завершения): ")
    num = 1
    while num != 0:
        num = int(input())
        if num != 0:
            print("Подвердите это действие\nНаберите 'удалить'")
            confirm = input()
            if confirm == "удалить":
                del_channels(num)
                print("канал удален")
            else:
                choice()
                return
    choice()

def opt_view():
    name = int(input("Введи номер нужного канала: ")) - 1
    history(name)
    choice()
    time.sleep(5)

def choice():
    print("Выберите действие;\n1.удаление\n2. просмотр\n")
    choice_num = int(input())
    channels = database.get_all_channels()
    n = 0
    for ch_id, ch_name in channels:
        n += 1
        print(f"{n}. {ch_name}")
    if choice_num == 1:
        opt_del()
    elif choice_num == 2:
        opt_view()

choice()



