
- Run fluent-bit container  
```sh
cd fluent-bit
docker-compose up -d
  ```
- make sure fluentd container is running and listenning in port 24224
-any logs inserted in logs file will be injected in analytics database