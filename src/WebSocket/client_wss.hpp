#ifndef SIMPLE_WEB_CLIENT_WSS_HPP
#define SIMPLE_WEB_CLIENT_WSS_HPP

#include "client_ws.hpp"

#ifdef ASIO_STANDALONE
#include <asio/ssl.hpp>
#else
#include <boost/asio/ssl.hpp>
#endif

namespace SimpleWeb {
  using WSS = asio::ssl::stream<asio::ip::tcp::socket>;

  template <>
  class SocketClient<WSS> : public SocketClientBase<WSS> {
  public:
    /**
     * Constructs a client object.
     *
     * @param server_port_path   Server resource given by host[:port][/path]
     * @param verify_certificate Set to true (default) to verify the server's certificate and hostname according to RFC 2818.
     * @param certification_file If non-empty, sends the given certification file to server. Requires private_key_file.
     * @param private_key_file   If non-empty, specifies the file containing the private key for certification_file. Requires certification_file.
     * @param verify_file        If non-empty, use this certificate authority file to perform verification.
     */
    SocketClient(const std::string &server_port_path, bool verify_certificate = true,
                 const std::string &certification_file = std::string(), const std::string &private_key_file = std::string(),
                 const std::string &verify_file = std::string())
        : SocketClientBase<WSS>::SocketClientBase(server_port_path, 443),
#if(defined(ASIO_STANDALONE) && ASIO_VERSION >= 101300) || BOOST_ASIO_VERSION >= 101300
          context(asio::ssl::context::tls_client) {
      // Disabling TLS 1.0 and 1.1 (see RFC 8996)
      context.set_options(asio::ssl::context::no_tlsv1);
      context.set_options(asio::ssl::context::no_tlsv1_1);
#else
          context(asio::ssl::context::tlsv12) {
#endif

      if(certification_file.size() > 0 && private_key_file.size() > 0) {
        context.use_certificate_chain_file(certification_file);
        context.use_private_key_file(private_key_file, asio::ssl::context::pem);
      }

      if(verify_certificate)
        context.set_verify_callback(asio::ssl::rfc2818_verification(host));

      if(verify_file.size() > 0)
        context.load_verify_file(verify_file);
      else
        context.set_default_verify_paths();

      if(verify_certificate)
        context.set_verify_mode(asio::ssl::verify_peer);
      else
        context.set_verify_mode(asio::ssl::verify_none);
    };

  protected:
    asio::ssl::context context;

    void connect() override {
      LockGuard connection_lock(connection_mutex);
      auto connection = this->connection = std::shared_ptr<Connection>(new Connection(handler_runner, config.timeout_idle, *io_service, context));
      connection_lock.unlock();

      std::pair<std::string, std::string> host_port;
      if(config.proxy_server.empty())
        host_port = {host, std::to_string(port)};
      else {
        auto proxy_host_port = parse_host_port(config.proxy_server, 8080);
        host_port = {proxy_host_port.first, std::to_string(proxy_host_port.second)};
      }

      auto resolver = std::make_shared<asio::ip::tcp::resolver>(*io_service);
      connection->set_timeout(config.timeout_request);
      async_resolve(*resolver, host_port, [this, connection, resolver](const error_code &ec, resolver_results results) {
        connection->cancel_timeout();
        auto lock = connection->handler_runner->continue_lock();
        if(!lock)
          return;
        if(!ec) {
          connection->set_timeout(this->config.timeout_request);
          asio::async_connect(connection->socket->lowest_layer(), results, [this, connection, resolver](const error_code &ec, async_connect_endpoint /*endpoint*/) {
            connection->cancel_timeout();
            auto lock = connection->handler_runner->continue_lock();
            if(!lock)
              return;
            if(!ec) {
              asio::ip::tcp::no_delay option(true);
              error_code ec;
              connection->socket->lowest_layer().set_option(option, ec);

              if(!this->config.proxy_server.empty()) {
                auto streambuf = std::make_shared<asio::streambuf>();
                std::ostream ostream(streambuf.get());
                auto host_port = this->host + ':' + std::to_string(this->port);
                ostream << "CONNECT " + host_port + " HTTP/1.1\r\n"
                        << "Host: " << host_port << "\r\n";
                if(!this->config.proxy_auth.empty())
                  ostream << "Proxy-Authorization: Basic " << Crypto::Base64::encode(this->config.proxy_auth) << "\r\n";
                ostream << "\r\n";
                connection->set_timeout(this->config.timeout_request);
                asio::async_write(connection->socket->next_layer(), *streambuf, [this, connection, streambuf](const error_code &ec, std::size_t /*bytes_transferred*/) {
                  connection->cancel_timeout();
                  auto lock = connection->handler_runner->continue_lock();
                  if(!lock)
                    return;
                  if(!ec) {
                    connection->set_timeout(this->config.timeout_request);
                    connection->in_message = std::shared_ptr<InMessage>(new InMessage());
                    asio::async_read_until(connection->socket->next_layer(), connection->in_message->streambuf, "\r\n\r\n", [this, connection](const error_code &ec, std::size_t /*bytes_transferred*/) {
                      connection->cancel_timeout();
                      auto lock = connection->handler_runner->continue_lock();
                      if(!lock)
                        return;
                      if(!ec) {
                        if(!ResponseMessage::parse(*connection->in_message, connection->http_version, connection->status_code, connection->header))
                          this->connection_error(connection, make_error_code::make_error_code(errc::protocol_error));
                        else {
                          if(connection->status_code.compare(0, 3, "200") != 0)
                            this->connection_error(connection, make_error_code::make_error_code(errc::permission_denied));
                          else
                            this->handshake(connection);
                        }
                      }
                      else
                        this->connection_error(connection, ec);
                    });
                  }
                  else
                    this->connection_error(connection, ec);
                });
              }
              else
                this->handshake(connection);
            }
            else
              this->connection_error(connection, ec);
          });
        }
        else
          this->connection_error(connection, ec);
      });
    }

    void handshake(const std::shared_ptr<Connection> &connection) {
      SSL_set_tlsext_host_name(connection->socket->native_handle(), this->host.c_str());

      connection->set_timeout(this->config.timeout_request);
      connection->socket->async_handshake(asio::ssl::stream_base::client, [this, connection](const error_code &ec) {
        connection->cancel_timeout();
        auto lock = connection->handler_runner->continue_lock();
        if(!lock)
          return;
        if(!ec)
          upgrade(connection);
        else
          this->connection_error(connection, ec);
      });
    }
  };
} // namespace SimpleWeb

#endif /* SIMPLE_WEB_CLIENT_WSS_HPP */
