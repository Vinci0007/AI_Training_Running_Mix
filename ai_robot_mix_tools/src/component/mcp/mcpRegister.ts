// 使用mcp SDK注册设备
import { MCP } from '@mcp/mcp-sdk';

const mcp = new MCP({
  apiKey: 'your_api_key',
  apiSecret: 'your_api_secret',
  host: 'https://api.mcp.com',
});

mcp.registerDevice({
  deviceType: 'robot',
  deviceName: 'your_device_name',
  deviceDescription: 'your_device_description',
  deviceTags: ['your_device_tags'],
  deviceCapabilities: ['your_device_capabilities'],
}).then((response) => {
  console.log(response);
}).catch((error) => {
  console.error(error);
});