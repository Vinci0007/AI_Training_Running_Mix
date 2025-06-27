"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.getRemoteServerInfo = exports.disconnectServer = exports.connectServer = void 0;
// 连接远程服务器
const child_process_1 = require("child_process");
function connectServer(ip, port, user, password) {
    const cmd = `sshpass -p ${password} ssh ${user}@${ip} -p ${port}`;
    (0, child_process_1.exec)(cmd, (error, stdout, stderr) => {
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
exports.connectServer = connectServer;
function disconnectServer(ip, port, user, password) {
    const cmd = `sshpass -p ${password} ssh ${user}@${ip} -p ${port} "exit"`;
    (0, child_process_1.exec)(cmd, (error, stdout, stderr) => {
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
exports.disconnectServer = disconnectServer;
function getRemoteServerInfo(ip, port, user, password) {
    const cmd = `sshpass -p ${password} ssh ${user}@${ip} -p ${port} "uname -a"`;
    (0, child_process_1.exec)(cmd, (error, stdout, stderr) => {
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
exports.getRemoteServerInfo = getRemoteServerInfo;
exports.default = connectServer;
