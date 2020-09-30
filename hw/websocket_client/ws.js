let addrs = ['192.168.0.x'];

class esp32
{
	constructor(addr, index)
	{
		this.last_cpu_time = 0;
		this.addr = addr;
		this.index = index;
		this.ws = new WebSocket('ws://' + addr);
		this.ws.onmessage = (msg) =>
		{
		  console.log(addr, index, msg.data);
		  if(msg.data.startsWith("CPU_Time: "))
		  {
		    this.last_cpu_time = parseInt(msg.data.substring(10));
		  }
		};
		this.ws.onclose = () => { console.log(addr, index, 'closed'); }
		this.ws.onerror = () => { console.log(addr, index, 'error'); }
	}
	run(delay, len = 60000)
	{
		this.ws.send('start_at ' + (this.last_cpu_time + delay) + ' ' + parseInt(len));
	}
	ntp(ntp_addr)
	{
		this.ws.send('ntp ' + ntp_addr);
	}
	cpu()
	{
		this.ws.send('cpu_time');
	}
	led_on()
	{
		this.ws.send('led_on');
	}
	led_off()
	{
		this.ws.send('led_off');
	}
	show_state()
	{
		this.ws.send('show_state');
	}
	set_state(name, value)
	{
		this.ws.send('set_state ' + name + ' ' + value);
	}
	format()
	{
		this.ws.send('format');
	}
	web()
	{
		this.ws.send('web_mode');
	}
}

esps = [];
for(let index in addrs)
{
	esps.push(new esp32(addrs[index], index));
}

// ---

for(let esp of esps)
{
	esp.ntp('192.168.0.y');
}

for(let esp of esps)
{
	esp.cpu();
}

for(let esp of esps)
{
	esp.run(5000);
}
