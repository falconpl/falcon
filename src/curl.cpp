/*
   FALCON - The Falcon Programming Language.
   FILE: curl_ext.cpp

   cURL library binding for Falcon
   Main module file, providing the module object to
   the Falcon engine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 27 Nov 2009 16:31:15 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: The above AUTHOR

         Licensed under the Falcon Programming Language License,
      Version 1.1 (the "License"); you may not use this file
      except in compliance with the License. You may obtain
      a copy of the License at

         http://www.falconpl.org/?page_id=license_1_1

      Unless required by applicable law or agreed to in writing,
      software distributed under the License is distributed on
      an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
      KIND, either express or implied. See the License for the
      specific language governing permissions and limitations
      under the License.

*/

/** \file
   Main module file, providing the module object to
   the Falcon engine.
*/

#include <curl/curl.h>
#include <falcon/module.h>
#include "curl_ext.h"
#include "curl_mod.h"
#include "curl_st.h"

#include "version.h"

//==================================================
// Extension of Falcon module
//==================================================

class CurlModule: public Falcon::Module
{
   static int init_count;

public:
   CurlModule();
   virtual ~CurlModule();
};


int CurlModule::init_count = 0;

CurlModule::CurlModule():
   Module()
{
   if( init_count == 0 )
   {
      curl_global_init( CURL_GLOBAL_ALL );
   }

   ++init_count;
}


CurlModule::~CurlModule()
{
   if( --init_count == 0 )
      curl_global_cleanup();
}

/*#
   @main curl

   This entry creates the main page of your module documentation.

   If your project will generate more modules, you may creaete a
   multi-module documentation by adding a module entry like the
   following

   @code
      \/*#
         \@module module_name Title of the module docs
         \@brief Brief description in module list..

         Some documentation...
      *\/
   @endcode

   And use the \@beginmodule <modulename> code at top of the _ext file
   (or files) where the extensions functions for that modules are
   documented.
*/

FALCON_MODULE_DECL
{
   #define FALCON_DECLARE_MODULE self

   // initialize the module
   Falcon::Module *self = new CurlModule();
   self->name( "curl" );
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   //============================================================
   // Here declare the international string table implementation
   //
   #include "curl_st.h"

   //============================================================
   // Here declare CURL - easy api
   //
   self->addExtFunc( "curl_version", Falcon::Ext::curl_version );

   Falcon::Symbol *easy_class = self->addClass( "Handle", Falcon::Ext::Handle_init )
      ->addParam( "uri" );
   easy_class->getClassDef()->factory( &Falcon::Mod::CurlHandle::Factory );
   self->addClassMethod( easy_class, "exec", Falcon::Ext::Handle_exec );

   self->addClassMethod( easy_class, "setOutConsole", Falcon::Ext::Handle_setOutConsole );
   self->addClassMethod( easy_class, "setOutString", Falcon::Ext::Handle_setOutString );
   self->addClassMethod( easy_class, "setOutStream", Falcon::Ext::Handle_setOutStream ).asSymbol()
      ->addParam( "stream" );
   self->addClassMethod( easy_class, "setOutCallback", Falcon::Ext::Handle_setOutCallback ).asSymbol()
      ->addParam( "cb" );
   self->addClassMethod( easy_class, "setInStream", Falcon::Ext::Handle_setInStream ).asSymbol()
      ->addParam( "stream" );
   self->addClassMethod( easy_class, "setInCallback", Falcon::Ext::Handle_setInCallback ).asSymbol()
      ->addParam( "cb" );

   //self->addClassMethod( easy_class, "setOutMessage", Falcon::Ext::Handle_setOutMessage ).asSymbol()
   //   ->addParam( "slot" );
   self->addClassMethod( easy_class, "getData", Falcon::Ext::Handle_getData );
   self->addClassMethod( easy_class, "setOption", Falcon::Ext::Handle_setOption ).asSymbol()
      ->addParam( "option" )->addParam( "data" );
   self->addClassMethod( easy_class, "postData", Falcon::Ext::Handle_postData ).asSymbol()
      ->addParam( "data" );

   self->addClassMethod( easy_class, "cleanup", Falcon::Ext::Handle_cleanup );

   //============================================================
   // Enumeration class CURL
   //
   Falcon::Symbol *curle_class = self->addClass( "CURL" );
   self->addClassProperty( curle_class, "READFUNC_ABORT" ).setInteger(CURL_READFUNC_ABORT).setReadOnly(true);
   self->addClassProperty( curle_class, "READFUNC_PAUSE" ).setInteger(CURL_READFUNC_PAUSE).setReadOnly(true);
   self->addClassProperty( curle_class, "WRITEFUNC_PAUSE" ).setInteger(CURL_WRITEFUNC_PAUSE).setReadOnly(true);

   //============================================================
   // Enumeration class COPT
   //
   Falcon::Symbol *copt_class = self->addClass( "CURLOPT" );
   self->addClassProperty( copt_class, "VERBOSE" ).setInteger(CURLOPT_VERBOSE).setReadOnly(true);
   self->addClassProperty( copt_class, "HEADER" ).setInteger(CURLOPT_HEADER).setReadOnly(true);
   self->addClassProperty( copt_class, "NOPROGRESS" ).setInteger(CURLOPT_NOPROGRESS).setReadOnly(true);

   self->addClassProperty( copt_class, "HTTPPROXYTUNNEL" ).setInteger(CURLOPT_HTTPPROXYTUNNEL).setReadOnly(true);
   self->addClassProperty( copt_class, "SOCKS5_GSSAPI_NEC" ).setInteger(CURLOPT_SOCKS5_GSSAPI_NEC).setReadOnly(true);
   self->addClassProperty( copt_class, "TCP_NODELAY" ).setInteger(CURLOPT_TCP_NODELAY).setReadOnly(true);
   self->addClassProperty( copt_class, "AUTOREFERER" ).setInteger(CURLOPT_AUTOREFERER).setReadOnly(true);
   self->addClassProperty( copt_class, "FOLLOWLOCATION" ).setInteger(CURLOPT_FOLLOWLOCATION).setReadOnly(true);
   self->addClassProperty( copt_class, "UNRESTRICTED_AUTH" ).setInteger(CURLOPT_UNRESTRICTED_AUTH).setReadOnly(true);
   self->addClassProperty( copt_class, "PUT" ).setInteger(CURLOPT_PUT).setReadOnly(true);
   self->addClassProperty( copt_class, "POST" ).setInteger(CURLOPT_POST).setReadOnly(true);
   self->addClassProperty( copt_class, "COOKIESESSION" ).setInteger(CURLOPT_COOKIESESSION).setReadOnly(true);
   self->addClassProperty( copt_class, "HTTPGET" ).setInteger(CURLOPT_HTTPGET).setReadOnly(true);
   self->addClassProperty( copt_class, "IGNORE_CONTENT_LENGTH" ).setInteger(CURLOPT_IGNORE_CONTENT_LENGTH).setReadOnly(true);
   self->addClassProperty( copt_class, "DIRLISTONLY" ).setInteger(CURLOPT_DIRLISTONLY).setReadOnly(true);
   self->addClassProperty( copt_class, "APPEND" ).setInteger(CURLOPT_APPEND).setReadOnly(true);
   self->addClassProperty( copt_class, "FTP_USE_EPRT" ).setInteger(CURLOPT_FTP_USE_EPRT).setReadOnly(true);
   self->addClassProperty( copt_class, "FTP_USE_EPSV" ).setInteger(CURLOPT_FTP_USE_EPSV).setReadOnly(true);
   self->addClassProperty( copt_class, "FTP_CREATE_MISSING_DIRS" ).setInteger(CURLOPT_FTP_CREATE_MISSING_DIRS).setReadOnly(true);
   self->addClassProperty( copt_class, "FTP_SKIP_PASV_IP" ).setInteger(CURLOPT_FTP_SKIP_PASV_IP).setReadOnly(true);
   self->addClassProperty( copt_class, "TRANSFERTEXT" ).setInteger(CURLOPT_TRANSFERTEXT).setReadOnly(true);
   self->addClassProperty( copt_class, "PROXY_TRANSFER_MODE" ).setInteger(CURLOPT_PROXY_TRANSFER_MODE).setReadOnly(true);
   self->addClassProperty( copt_class, "CRLF" ).setInteger(CURLOPT_CRLF).setReadOnly(true);
   self->addClassProperty( copt_class, "FILETIME" ).setInteger(CURLOPT_FILETIME).setReadOnly(true);
   self->addClassProperty( copt_class, "NOBODY" ).setInteger(CURLOPT_NOBODY).setReadOnly(true);
   self->addClassProperty( copt_class, "UPLOAD" ).setInteger(CURLOPT_UPLOAD).setReadOnly(true);
   self->addClassProperty( copt_class, "FRESH_CONNECT" ).setInteger(CURLOPT_FRESH_CONNECT).setReadOnly(true);
   self->addClassProperty( copt_class, "FORBID_REUSE" ).setInteger(CURLOPT_FORBID_REUSE).setReadOnly(true);
   self->addClassProperty( copt_class, "CONNECT_ONLY" ).setInteger(CURLOPT_CONNECT_ONLY).setReadOnly(true);
   self->addClassProperty( copt_class, "SSLENGINE_DEFAULT" ).setInteger(CURLOPT_SSLENGINE_DEFAULT).setReadOnly(true);
   self->addClassProperty( copt_class, "SSL_VERIFYPEER" ).setInteger(CURLOPT_SSL_VERIFYPEER).setReadOnly(true);
   self->addClassProperty( copt_class, "CERTINFO" ).setInteger(CURLOPT_CERTINFO).setReadOnly(true);
   self->addClassProperty( copt_class, "SSL_VERIFYHOST" ).setInteger(CURLOPT_SSL_VERIFYHOST).setReadOnly(true);
   self->addClassProperty( copt_class, "SSL_SESSIONID_CACHE" ).setInteger(CURLOPT_SSL_SESSIONID_CACHE).setReadOnly(true);

   self->addClassProperty( copt_class, "PROTOCOLS" ).setInteger(CURLOPT_PROTOCOLS).setReadOnly(true);
   self->addClassProperty( copt_class, "REDIR_PROTOCOLS" ).setInteger(CURLOPT_REDIR_PROTOCOLS).setReadOnly(true);
   self->addClassProperty( copt_class, "PROXYPORT" ).setInteger(CURLOPT_PROXYPORT).setReadOnly(true);
   self->addClassProperty( copt_class, "PROXYTYPE" ).setInteger(CURLOPT_PROXYTYPE).setReadOnly(true);


   Falcon::Symbol *cproxy_class = self->addClass( "CURLPROXY" );
   self->addClassProperty( cproxy_class, "HTTP" ).setInteger(CURLPROXY_HTTP).setReadOnly(true);
   self->addClassProperty( cproxy_class, "HTTP_1_0" ).setInteger(CURLPROXY_HTTP_1_0).setReadOnly(true);
   self->addClassProperty( cproxy_class, "SOCKS4" ).setInteger(CURLPROXY_SOCKS4).setReadOnly(true);
   self->addClassProperty( cproxy_class, "SOCKS5" ).setInteger(CURLPROXY_SOCKS5).setReadOnly(true);
   self->addClassProperty( cproxy_class, "SOCKS4A" ).setInteger(CURLPROXY_SOCKS4A).setReadOnly(true);

   self->addClassProperty( copt_class, "LOCALPORT" ).setInteger(CURLOPT_LOCALPORT).setReadOnly(true);
   self->addClassProperty( copt_class, "LOCALPORTRANGE" ).setInteger(CURLOPT_LOCALPORTRANGE).setReadOnly(true);
   self->addClassProperty( copt_class, "DNS_CACHE_TIMEOUT" ).setInteger(CURLOPT_DNS_CACHE_TIMEOUT).setReadOnly(true);
   self->addClassProperty( copt_class, "DNS_USE_GLOBAL_CACHE" ).setInteger(CURLOPT_DNS_USE_GLOBAL_CACHE).setReadOnly(true);
   self->addClassProperty( copt_class, "BUFFERSIZE" ).setInteger(CURLOPT_BUFFERSIZE).setReadOnly(true);
   self->addClassProperty( copt_class, "PORT" ).setInteger(CURLOPT_PORT).setReadOnly(true);
   self->addClassProperty( copt_class, "ADDRESS_SCOPE" ).setInteger(CURLOPT_ADDRESS_SCOPE).setReadOnly(true);
   self->addClassProperty( copt_class, "NETRC" ).setInteger(CURLOPT_NETRC).setReadOnly(true);


   Falcon::Symbol *cnetrc_class = self->addClass( "CURL_NETRC" );
   self->addClassProperty( cnetrc_class, "OPTIONAL" ).setInteger(CURL_NETRC_OPTIONAL).setReadOnly(true);
   self->addClassProperty( cnetrc_class, "IGNORED" ).setInteger(CURL_NETRC_IGNORED).setReadOnly(true);

   self->addClassProperty( copt_class, "HTTPAUTH" ).setInteger(CURLOPT_HTTPAUTH).setReadOnly(true);

   Falcon::Symbol *cauth_class = self->addClass( "CURLAUTH" );
   self->addClassProperty( cauth_class, "BASIC" ).setInteger(CURLAUTH_BASIC).setReadOnly(true);
   self->addClassProperty( cauth_class, "DIGEST" ).setInteger(CURLAUTH_DIGEST).setReadOnly(true);
   self->addClassProperty( cauth_class, "DIGEST_IE" ).setInteger(CURLAUTH_DIGEST_IE).setReadOnly(true);
   self->addClassProperty( cauth_class, "GSSNEGOTIATE" ).setInteger(CURLAUTH_GSSNEGOTIATE).setReadOnly(true);
   self->addClassProperty( cauth_class, "NTLM" ).setInteger(CURLAUTH_NTLM).setReadOnly(true);
   self->addClassProperty( cauth_class, "ANY" ).setInteger(CURLAUTH_ANY).setReadOnly(true);
   self->addClassProperty( cauth_class, "ANYSAFE" ).setInteger(CURLAUTH_ANYSAFE).setReadOnly(true);

   self->addClassProperty( copt_class, "PROXYAUTH" ).setInteger(CURLOPT_PROXYAUTH).setReadOnly(true);
   self->addClassProperty( copt_class, "MAXREDIRS" ).setInteger(CURLOPT_MAXREDIRS).setReadOnly(true);
   self->addClassProperty( copt_class, "POSTREDIR" ).setInteger(CURLOPT_POSTREDIR).setReadOnly(true);
   self->addClassProperty( copt_class, "HTTP_VERSION" ).setInteger(CURLOPT_HTTP_VERSION).setReadOnly(true);

   Falcon::Symbol *chttp_class = self->addClass( "CURL_HTTP" );
   self->addClassProperty( chttp_class, "VERSION_NONE" ).setInteger(CURL_HTTP_VERSION_NONE).setReadOnly(true);
   self->addClassProperty( chttp_class, "VERSION_1_0" ).setInteger(CURL_HTTP_VERSION_1_0).setReadOnly(true);
   self->addClassProperty( chttp_class, "VERSION_1_1" ).setInteger(CURL_HTTP_VERSION_1_1).setReadOnly(true);

   self->addClassProperty( copt_class, "HTTP_CONTENT_DECODING" ).setInteger(CURLOPT_HTTP_CONTENT_DECODING).setReadOnly(true);
   self->addClassProperty( copt_class, "HTTP_TRANSFER_DECODING" ).setInteger(CURLOPT_HTTP_TRANSFER_DECODING).setReadOnly(true);
   self->addClassProperty( copt_class, "TFTP_BLKSIZE" ).setInteger(CURLOPT_TFTP_BLKSIZE).setReadOnly(true);
   self->addClassProperty( copt_class, "FTP_RESPONSE_TIMEOUT" ).setInteger(CURLOPT_FTP_RESPONSE_TIMEOUT).setReadOnly(true);
   self->addClassProperty( copt_class, "USE_SSL" ).setInteger(CURLOPT_USE_SSL).setReadOnly(true);

   Falcon::Symbol *cusessl_class = self->addClass( "CURL_USESSL" );
   self->addClassProperty( cusessl_class, "NONE" ).setInteger(CURLUSESSL_NONE).setReadOnly(true);
   self->addClassProperty( cusessl_class, "TRY" ).setInteger(CURLUSESSL_TRY).setReadOnly(true);
   self->addClassProperty( cusessl_class, "CONTROL" ).setInteger(CURLUSESSL_CONTROL).setReadOnly(true);
   self->addClassProperty( cusessl_class, "ALL" ).setInteger(CURLUSESSL_ALL).setReadOnly(true);

   self->addClassProperty( copt_class, "FTPSSLAUTH" ).setInteger(CURLOPT_FTPSSLAUTH).setReadOnly(true);

   Falcon::Symbol *cftpauth_class = self->addClass( "CURLFTPAUTH" );
   self->addClassProperty( cftpauth_class, "DEFAULT" ).setInteger(CURLFTPAUTH_DEFAULT).setReadOnly(true);
   self->addClassProperty( cftpauth_class, "SSL" ).setInteger(CURLFTPAUTH_SSL).setReadOnly(true);
   self->addClassProperty( cftpauth_class, "TLS" ).setInteger(CURLFTPAUTH_TLS).setReadOnly(true);

   self->addClassProperty( copt_class, "FTP_SSL_CCC" ).setInteger(CURLOPT_FTP_SSL_CCC).setReadOnly(true);

   Falcon::Symbol *cftpssl_ccc_class = self->addClass( "CURLFTPSSL_CCC" );
   self->addClassProperty( cftpssl_ccc_class, "NONE" ).setInteger(CURLFTPSSL_CCC_NONE).setReadOnly(true);
   self->addClassProperty( cftpssl_ccc_class, "PASSIVE" ).setInteger(CURLFTPSSL_CCC_PASSIVE).setReadOnly(true);
   self->addClassProperty( cftpssl_ccc_class, "ACTIVE" ).setInteger(CURLFTPSSL_CCC_ACTIVE).setReadOnly(true);


   self->addClassProperty( copt_class, "FTP_FILEMETHOD" ).setInteger(CURLOPT_FTP_FILEMETHOD).setReadOnly(true);

   Falcon::Symbol *cftpmethod_class = self->addClass( "CURLFTPMETHOD" );
   self->addClassProperty( cftpmethod_class, "MULTICWD" ).setInteger(CURLFTPMETHOD_MULTICWD).setReadOnly(true);
   self->addClassProperty( cftpmethod_class, "NOCWD" ).setInteger(CURLFTPSSL_CCC_PASSIVE).setReadOnly(true);
   self->addClassProperty( cftpmethod_class, "SINGLECWD" ).setInteger(CURLFTPMETHOD_SINGLECWD).setReadOnly(true);

   self->addClassProperty( copt_class, "RESUME_FROM" ).setInteger(CURLOPT_RESUME_FROM).setReadOnly(true);
   self->addClassProperty( copt_class, "INFILESIZE" ).setInteger(CURLOPT_INFILESIZE).setReadOnly(true);
   self->addClassProperty( copt_class, "MAXFILESIZE" ).setInteger(CURLOPT_MAXFILESIZE).setReadOnly(true);
   self->addClassProperty( copt_class, "TIMECONDITION" ).setInteger(CURLOPT_TIMECONDITION).setReadOnly(true);
   self->addClassProperty( copt_class, "TIMEVALUE" ).setInteger(CURLOPT_TIMEVALUE).setReadOnly(true);
   self->addClassProperty( copt_class, "TIMEOUT" ).setInteger(CURLOPT_TIMEOUT).setReadOnly(true);
   self->addClassProperty( copt_class, "TIMEOUT_MS" ).setInteger(CURLOPT_TIMEOUT_MS).setReadOnly(true);
   self->addClassProperty( copt_class, "LOW_SPEED_LIMIT" ).setInteger(CURLOPT_LOW_SPEED_LIMIT).setReadOnly(true);
   self->addClassProperty( copt_class, "LOW_SPEED_TIME" ).setInteger(CURLOPT_LOW_SPEED_TIME).setReadOnly(true);
   self->addClassProperty( copt_class, "MAXCONNECTS" ).setInteger(CURLOPT_MAXCONNECTS).setReadOnly(true);
   self->addClassProperty( copt_class, "CONNECTTIMEOUT" ).setInteger(CURLOPT_CONNECTTIMEOUT).setReadOnly(true);
   self->addClassProperty( copt_class, "CONNECTTIMEOUT_MS" ).setInteger(CURLOPT_CONNECTTIMEOUT_MS).setReadOnly(true);
   self->addClassProperty( copt_class, "IPRESOLVE" ).setInteger(CURLOPT_IPRESOLVE).setReadOnly(true);

   Falcon::Symbol *cipresolve_class = self->addClass( "CURL_IPRESOLVE" );
   self->addClassProperty( cipresolve_class, "WHATEVER" ).setInteger(CURL_IPRESOLVE_WHATEVER).setReadOnly(true);
   self->addClassProperty( cipresolve_class, "V4" ).setInteger(CURL_IPRESOLVE_V4).setReadOnly(true);
   self->addClassProperty( cipresolve_class, "V6" ).setInteger(CURL_IPRESOLVE_V6).setReadOnly(true);


   self->addClassProperty( copt_class, "SSLVERSION" ).setInteger(CURLOPT_SSLVERSION).setReadOnly(true);

   Falcon::Symbol *csslversion_class = self->addClass( "CURL_SSLVERSION" );
   self->addClassProperty( csslversion_class, "DEFAULT" ).setInteger(CURL_SSLVERSION_DEFAULT).setReadOnly(true);
   self->addClassProperty( csslversion_class, "TLSv1" ).setInteger(CURL_SSLVERSION_TLSv1).setReadOnly(true);
   self->addClassProperty( csslversion_class, "SSLv2" ).setInteger(CURL_SSLVERSION_SSLv2).setReadOnly(true);
   self->addClassProperty( csslversion_class, "SSLv3" ).setInteger(CURL_SSLVERSION_SSLv3).setReadOnly(true);

   self->addClassProperty( copt_class, "SSH_AUTH_TYPES" ).setInteger(CURLOPT_SSH_AUTH_TYPES).setReadOnly(true);

   Falcon::Symbol *cssh_auth_class = self->addClass( "CURLSSH_AUTH" );
   self->addClassProperty( cssh_auth_class, "PUBLICKEY" ).setInteger(CURLSSH_AUTH_PUBLICKEY).setReadOnly(true);
   self->addClassProperty( cssh_auth_class, "PASSWORD" ).setInteger(CURLSSH_AUTH_PASSWORD).setReadOnly(true);
   self->addClassProperty( cssh_auth_class, "HOST" ).setInteger(CURLSSH_AUTH_HOST).setReadOnly(true);
   self->addClassProperty( cssh_auth_class, "KEYBOARD" ).setInteger(CURLSSH_AUTH_KEYBOARD).setReadOnly(true);
   self->addClassProperty( cssh_auth_class, "ANY" ).setInteger(CURLSSH_AUTH_ANY).setReadOnly(true);


   self->addClassProperty( copt_class, "NEW_FILE_PERMS" ).setInteger(CURLOPT_NEW_FILE_PERMS).setReadOnly(true);
   self->addClassProperty( copt_class, "NEW_DIRECTORY_PERMS" ).setInteger(CURLOPT_NEW_DIRECTORY_PERMS).setReadOnly(true);

   self->addClassProperty( copt_class, "RESUME_FROM_LARGE" ).setInteger(CURLOPT_RESUME_FROM_LARGE).setReadOnly(true);
   self->addClassProperty( copt_class, "INFILESIZE_LARGE" ).setInteger(CURLOPT_INFILESIZE_LARGE).setReadOnly(true);
   self->addClassProperty( copt_class, "MAXFILESIZE_LARGE" ).setInteger(CURLOPT_MAXFILESIZE_LARGE).setReadOnly(true);
   self->addClassProperty( copt_class, "MAX_SEND_SPEED_LARGE" ).setInteger(CURLOPT_MAX_SEND_SPEED_LARGE).setReadOnly(true);
   self->addClassProperty( copt_class, "MAX_RECV_SPEED_LARGE" ).setInteger(CURLOPT_MAX_RECV_SPEED_LARGE).setReadOnly(true);

   self->addClassProperty( copt_class, "URL" ).setInteger(CURLOPT_URL).setReadOnly(true);
   self->addClassProperty( copt_class, "PROXY" ).setInteger(CURLOPT_PROXY).setReadOnly(true);
   self->addClassProperty( copt_class, "NOPROXY" ).setInteger(CURLOPT_NOPROXY).setReadOnly(true);
   self->addClassProperty( copt_class, "SOCKS5_GSSAPI_SERVICE" ).setInteger(CURLOPT_SOCKS5_GSSAPI_SERVICE).setReadOnly(true);
   self->addClassProperty( copt_class, "INTERFACE" ).setInteger(CURLOPT_INTERFACE).setReadOnly(true);
   self->addClassProperty( copt_class, "NETRC_FILE" ).setInteger(CURLOPT_NETRC_FILE).setReadOnly(true);
   self->addClassProperty( copt_class, "USERPWD" ).setInteger(CURLOPT_USERPWD).setReadOnly(true);
   self->addClassProperty( copt_class, "PROXYUSERPWD" ).setInteger(CURLOPT_PROXYUSERPWD).setReadOnly(true);
   self->addClassProperty( copt_class, "USERNAME" ).setInteger(CURLOPT_USERNAME).setReadOnly(true);
   self->addClassProperty( copt_class, "PASSWORD" ).setInteger(CURLOPT_PASSWORD).setReadOnly(true);

   self->addClassProperty( copt_class, "PROXYUSERNAME" ).setInteger(CURLOPT_PROXYUSERNAME).setReadOnly(true);
   self->addClassProperty( copt_class, "PROXYPASSWORD" ).setInteger(CURLOPT_PROXYPASSWORD).setReadOnly(true);
   self->addClassProperty( copt_class, "ENCODING" ).setInteger(CURLOPT_ENCODING).setReadOnly(true);
   self->addClassProperty( copt_class, "REFERER" ).setInteger(CURLOPT_REFERER).setReadOnly(true);
   self->addClassProperty( copt_class, "USERAGENT" ).setInteger(CURLOPT_USERAGENT).setReadOnly(true);
   self->addClassProperty( copt_class, "COOKIE" ).setInteger(CURLOPT_COOKIE).setReadOnly(true);
   self->addClassProperty( copt_class, "COOKIEFILE" ).setInteger(CURLOPT_COOKIEFILE).setReadOnly(true);
   self->addClassProperty( copt_class, "COOKIEJAR" ).setInteger(CURLOPT_COOKIEJAR).setReadOnly(true);
   self->addClassProperty( copt_class, "COOKIELIST" ).setInteger(CURLOPT_COOKIELIST).setReadOnly(true);
   self->addClassProperty( copt_class, "FTPPORT" ).setInteger(CURLOPT_FTPPORT).setReadOnly(true);
   self->addClassProperty( copt_class, "FTP_ALTERNATIVE_TO_USER" ).setInteger(CURLOPT_FTP_ALTERNATIVE_TO_USER).setReadOnly(true);
   self->addClassProperty( copt_class, "FTP_ACCOUNT" ).setInteger(CURLOPT_FTP_ACCOUNT).setReadOnly(true);
   self->addClassProperty( copt_class, "RANGE" ).setInteger(CURLOPT_RANGE).setReadOnly(true);
   self->addClassProperty( copt_class, "CUSTOMREQUEST" ).setInteger(CURLOPT_CUSTOMREQUEST).setReadOnly(true);
   self->addClassProperty( copt_class, "SSLCERT" ).setInteger(CURLOPT_SSLCERT).setReadOnly(true);
   self->addClassProperty( copt_class, "SSLCERTTYPE" ).setInteger(CURLOPT_SSLCERTTYPE).setReadOnly(true);
   self->addClassProperty( copt_class, "SSLKEY" ).setInteger(CURLOPT_SSLKEY).setReadOnly(true);
   self->addClassProperty( copt_class, "SSLKEYTYPE" ).setInteger(CURLOPT_SSLKEYTYPE).setReadOnly(true);
   self->addClassProperty( copt_class, "KEYPASSWD" ).setInteger(CURLOPT_KEYPASSWD).setReadOnly(true);
   self->addClassProperty( copt_class, "SSLENGINE" ).setInteger(CURLOPT_SSLENGINE).setReadOnly(true);
   self->addClassProperty( copt_class, "CAINFO" ).setInteger(CURLOPT_CAINFO).setReadOnly(true);
   self->addClassProperty( copt_class, "ISSUERCERT" ).setInteger(CURLOPT_ISSUERCERT).setReadOnly(true);
   self->addClassProperty( copt_class, "CAPATH" ).setInteger(CURLOPT_CAPATH).setReadOnly(true);
   self->addClassProperty( copt_class, "CRLFILE" ).setInteger(CURLOPT_CRLFILE).setReadOnly(true);
   self->addClassProperty( copt_class, "RANDOM_FILE" ).setInteger(CURLOPT_RANDOM_FILE).setReadOnly(true);
   self->addClassProperty( copt_class, "EGDSOCKET" ).setInteger(CURLOPT_EGDSOCKET).setReadOnly(true);
   self->addClassProperty( copt_class, "SSL_CIPHER_LIST" ).setInteger(CURLOPT_SSL_CIPHER_LIST).setReadOnly(true);
   self->addClassProperty( copt_class, "KRBLEVEL" ).setInteger(CURLOPT_KRBLEVEL).setReadOnly(true);
   self->addClassProperty( copt_class, "SSH_HOST_PUBLIC_KEY_MD5" ).setInteger(CURLOPT_SSH_HOST_PUBLIC_KEY_MD5).setReadOnly(true);
   self->addClassProperty( copt_class, "SSH_PUBLIC_KEYFILE" ).setInteger(CURLOPT_SSH_PUBLIC_KEYFILE).setReadOnly(true);
   self->addClassProperty( copt_class, "SSH_PRIVATE_KEYFILE" ).setInteger(CURLOPT_SSH_PRIVATE_KEYFILE).setReadOnly(true);

#ifdef CURLOPT_SSH_KNOWNHOSTS
   self->addClassProperty( copt_class, "SSH_KNOWNHOSTS" ).setInteger(CURLOPT_SSH_KNOWNHOSTS).setReadOnly(true);
#endif

   // List options
   self->addClassProperty( copt_class, "HTTPHEADER" ).setInteger(CURLOPT_HTTPHEADER).setReadOnly(true);
   self->addClassProperty( copt_class, "HTTP200ALIASES" ).setInteger(CURLOPT_HTTP200ALIASES).setReadOnly(true);
   self->addClassProperty( copt_class, "QUOTE" ).setInteger(CURLOPT_QUOTE).setReadOnly(true);
   self->addClassProperty( copt_class, "POSTQUOTE" ).setInteger(CURLOPT_POSTQUOTE).setReadOnly(true);
   self->addClassProperty( copt_class, "PREQUOTE" ).setInteger(CURLOPT_PREQUOTE).setReadOnly(true);

   // To be implemented separately
   /*
   CURLOPT_HTTPPOST

   CURLOPT_SSH_KEYFUNCTION
   CURLOPT_SSH_KEYDATA

   CURLOPT_SHARE (?)

   CURLOPT_TELNETOPTIONS
   */

   //============================================================
   // CurlError class
   Falcon::Symbol *error_class = self->addExternalRef( "Error" ); // it's external
   Falcon::Symbol *ce_cls = self->addClass( "CurlError", Falcon::Ext::CurlError_init );
   ce_cls->setWKS( true );
   ce_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );


   return self;
}

/* end of curl.cpp */
