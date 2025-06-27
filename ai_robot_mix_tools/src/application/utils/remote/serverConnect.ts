// 连接远程服务器
import { exec } from 'child_process';

export function connectServer(ip: string, port: string, user: string, password: string) {
  const cmd = `sshpass -p ${password} ssh ${user}@${ip} -p ${port}`;
  exec(cmd, (error, stdout, stderr) => {
    if (error) {
      console.log(`error: ${error.message}`);
      return;
    }
    if (stderr) {
      console.log(`stderr: ${stderr}`);
      return;
    }
    console.log(`stdout: ${stdout}`);
  });
}

export function disconnectServer(ip: string, port: string, user: string, password: string) {
  const cmd = `sshpass -p ${password} ssh ${user}@${ip} -p ${port} "exit"`;
  exec(cmd, (error, stdout, stderr) => {
    if (error) {
      console.log(`error: ${error.message}`);
      return;
    }
    if (stderr) {
      console.log(`stderr: ${stderr}`);
      return;
    }
    console.log(`stdout: ${stdout}`);
  });
}

export function getRemoteServerInfo(ip: string, port: string, user: string, password: string) {
  const cmd = `sshpass -p ${password} ssh ${user}@${ip} -p ${port} "uname -a"`;
  exec(cmd, (error, stdout, stderr) => {
    if (error) {
      console.log(`error: ${error.message}`);
      return;
    }
    if (stderr) {
      console.log(`stderr: ${stderr}`);
      // get remote server info details
      console.warn(`Getting remote server system info .......`);
      // 
      return;
    }
    console.log(`stdout: ${stdout}`);
  });
}

export default connectServer;