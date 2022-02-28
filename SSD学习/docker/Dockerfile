FROM python:3.6-stretch

WORKDIR /tmp
COPY requirements.txt .


RUN pip install --upgrade pip
RUN pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple
RUN pip install -r requirements.txt

EXPOSE 5000

WORKDIR /app
COPY start.sh /home/start.sh
RUN chmod a+x /home/start.sh
CMD ["/home/start.sh"]
