---
title: How to set up your home server
tags: Projects
published: true
---
## How to set up your home server
A hosting Server is expensive, and it is not good for private users. However, we can run your personal computer as a home server with a few knowledges.

If your internet line is from IPS (e.g. AT&T), the public IP address is assigned to your router. By using this public IP, people can connect to your home network, but your router blocks all connections for security. Moreover, Routers make a sub-network which makes a network partition.  
However, there are several ways to communicate with computers over network partition.

1. Port Forward: To establish a socket communication, source IP address and port number / destination IP address and port number are required. But a computer which is on outside of network partition cannot get a server's address and port number since it only can get a router's address and port number. But if the router delivers all packets which come into an incoming port to inside address and port number, a computer from outside of network partition can communicate with a server. In this example your internal server ip is 192.168.0.1 and port number is 80 and incoming port of router is 8000.

	1. To enable port forwarding, go to router's management page and check that there is an option which is named with "NAT" or "Firewall". if you find it, there should be adding rule button. By using that you can add a rule which makes packets which come into 8000 go to 192.168.0.1:80 which is internal IP address.

	2. Re-start your router if you need and check the connection to the server with your router's ip and port number.

2. DMZ(Demilitarized zone) : a router with this method can make a Zone for internal servers. Servers in this zone cannot communicate with other computers which are in the network. In other words, they can communicate only with outside computers from network partition. Moreover, the router's firewall does not protect them. This means that the packet from outside of network partition can pass firewall. If you want to make server communicate with a computer from outside, this option should be applied to the server. Moreover, most routers with this option, they do port forwarding all incoming ports to the inside server.

	3. Find firewall or DMZ option in your router's management menu and add the server's ip to DMZ rule.
