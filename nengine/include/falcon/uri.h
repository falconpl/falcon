/*
   FALCON - The Falcon Programming Language.
   FILE: uri.h

   RFC 3986 - Uniform Resource Identifier
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Feb 2008 12:23:28 +0100

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_URI_H
#define FALCON_URI_H

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/path.h>
#include <falcon/enumerator.h>

namespace Falcon
{
class QueryMap;

/** RFC 3986 - Uniform Resource Identifier.

   This class offer falcon engine and its users an interface to URI.
*/
class FALCON_DYN_CLASS URI
{
public:
   
   /** URI Authority.
    This class is used to break an authority field into its components,
    and to easily manage it to set it back in an encoded form.
    */
   class FALCON_DYN_CLASS Authority
   {
   public:      
      /** Creates an autority with all the fields empty. */
      Authority()
      {}
      
      /** Creates an autority filling all the fields. 
       \param host The host responding to this authority.
       \param port the port where this autority responds.
       \param user User to be connected to the authority.
       \param pwd Password to be used.
       */
      Authority( const String& host, const String& port, const String& user, const String& pwd ):
         m_user( user ),
         m_password( pwd ),
         m_host( host ),
         m_port( port ),
         m_encoded("@")
      {}
      
      Authority( const Authority& other ):
         m_user( other.m_user ),
         m_password( other.m_password ),
         m_host( other.m_host ),
         m_port( other.m_port ),
         m_encoded( other.m_encoded )      
      {}
      
      /** The user part of this authority. */
      const String& user() const { return m_user; }
      /** The password part of this authority. */
      const String& password() const { return m_password; }
      /** The host part of this authority. */
      const String& host() const { return m_host; }
      /** The port part of this authority. */
      const String& port() const { return m_port; }

      /** Changes the user part of this authority. */
      void user( const String& value ) { m_encoded = "@"; m_user = value; }
      /** The password part of this authority. */
      void password( const String& value  ) { m_encoded = "@"; m_password = value; }
      /** The host part of this authority. */
      void host( const String& value  ) { m_encoded = "@"; m_host = value; }
      /** The port part of this authority. */
      void port( const String& value  ) { m_encoded = "@"; m_port = value; }
      
     /** Parse an authority string.
      \param uriAuth The source string considered an authority.
      \return true on success, false if the authority could not be parsed.
      */       
     bool parse( const String& uriAuth );
     
     /** Returns the encoded elements of this authority. */
     const String& encode() const;
     
   private:
      String m_user;
      String m_password;
      String m_host;
      String m_port;

      mutable String m_encoded;
   };
   
   
   /** Class used to access the quwery part of an URI.
    */
   class FALCON_DYN_CLASS Query
   {
   public:
      Query();
      ~Query();
      
      /** Parse an authority string.
      \param uriAuth The source string considered an authority.
      */    
      bool parse( const String& query );

      /** Returns the encoded elements of this authority. */
      const String& encode() const;

      /** Returns the number of keys in this querty. */
      size_t size() const;

      /** Returns the number of keys in this query. */
      void put( const String& field, const String& value );

      /** Returns the number of keys in this query. */
      bool get( const String& field, String& value ) const;

      /** Removes a query field. 
       \return True if the key existed, false otherwise.
       */
      bool remove( const String &key );
      
      /** Removes all the keys in the query.
       */
      void clear();

      /** Class used to enumerate key/value pairs.
       */
      class KeyValue
      {
      public:
         const String& key;
         const String& value;

         KeyValue( const String& k, const String& v ):
            key(k), value(v)
            {}
      };

      typedef Enumerator<KeyValue> FieldEnumerator;

      /** Enumerates the query fields - gets the first field.
         Calls back the given enumerator filling it with the required data.
      */
      void enumerateFields( FieldEnumerator& etor ) const; 
     
      private:
         class Private;
         Private* _p;

         mutable  String m_encoded;
      };

   /** Empty constructor.
      Creates an empty URI. To be filled at a later time.
   */
   URI();

   /** Complete URI constructor.
      Decodes the uri and parses it in its fields.
      \param suri The URI to be decoded.
      \param auth he URI::Authority part where to store the decoded autority fields
            or 0 if not necessary.
      \param path the Path part where to store the decoded path fields
            or 0 if not necessary.
      \param query the URI::Query part where to store the decoded query fields
            or 0 if not necessary.

      In case URI is not valid, isValid() will return false after construction.
    \note if any of auth, path or query parameters are given, well-formation
    tests will be performed and validity will be granted only if all the required
    parts can be properly parsed.
   */
   URI( const String &suri, Authority* auth = 0, Path* path = 0, Query* query = 0 )
   {
      m_bValid = parse( suri, auth, path, query );
   }

   /** Copy constructor.
      Copies everything in the other URI, including validity.
   */
   URI( const URI &other );

   virtual ~URI();

   /** Parses the given string into this URI.
      The URI will be normalized and eventually decoded, so that the internal
      format of the URI.

      Normally, the method decodes any % code into it's value, and considers
      the uri as encoded into UTF-8 sequences.

      If the \b decode param is false, the input string is read as-is.

      By default, the function will just store the query field for later retrival
      with the query() accessor. The query field will be returned in its original
      form, undecoded. If the makeQueryMap boolean field is set to true,
      the parseQuery() method will be called upon succesful completion of URI parsing,
      before the function returns.

      \param newUri the new URI to be parsed.     
      \return true on success, false if the given string is not a valid URI.
   */
   bool parse( const String &newUri, Authority* auth = 0, Path* path = 0, Query* query = 0 );

   
   /** Normalzies the URI sequence.
      Transforms ranges of ALPHA
      (%41-%5A and %61-%7A), DIGIT (%30-%39), hyphen (%2D), period (%2E),
      underscore (%5F), or tilde (%7E) into their coresponding values.

      Other than that, it transforms %20 in spaces.

      The normalization is performed on a result string.

      Every string set in input by any method in this class is normalized
      prior to storage.

      However, notice that parts in the class may be invalid URI elements
      if extracted as is, as they are stored as Falcon international strings.
      Encoding to URI conformance happens only when required.

      \param part the element of the URI (or the complete URI) to be normalized.
      \param result the string where the normalization is performed.
   */
   //static void normalize( const String &part, String &result );

   /** Returns current scheme. 
    \note The returned string is still URL encoded.
    */
   const String &scheme() const { return m_scheme; }

   /** Sets a different scheme for this URI.
    \param value A valid (and already escaped) scheme.
      This will invalidate current URI until next uri() is called.
   */
   void scheme( const String &value ) { m_encoded = "@"; m_scheme = value; }

   /** Returns the current path.
    \note The returned string is still URL encoded.
    */
   const String& auth() const { return m_authority; }
   
   /** Sets a different path for this URI.
      \param value A valid (and already escaped) uri path.
      This will invalidate current URI until next uri() is called.
   */
   void auth( const String &value ) { m_encoded = "@"; m_authority = value; }
   
   
   /** Returns the current path.
    \note The returned string is still URL encoded.
    */
   const String& path() const { return m_path; }
   
   /** Sets a different path for this URI.
      \param value A valid (and already escaped) uri path.
      This will invalidate current URI until next uri() is called.
   */
   void path( const String &value ) { m_encoded = "@"; m_path = value; }
   
   /** Returns the current query string.
    \note The returned string is still URL encoded.
    */
   const String& query() const { return m_query; }
   
   /** Sets a different path for this URI.
      \param value A valid (and already escaped) uri query string.
      This will invalidate current URI until next uri() is called.
      
    \note This method may corrupt the validity of the URI, if not properly
    used. Consider using URIQuery class to generate correctly formatted 
    qery strings.
   */
   void query( const String &value ) { m_encoded = "@"; m_query = value; }
   
   
   /** Returns the fragment part. */
   const String &fragment() const { return m_fragment; }

   /** Sets the fragment part. 
    \param value A valid (and already escaped) uri path.
      This will invalidate current URI until next encode() is called.
    */
   void fragment( const String &value ) { m_encoded = "@"; m_query = value; }

   /** Clears the content of this URI */
   void clear();
   
   /** Encode the URI. 
    \return The encoded contents of this URI.
    */
   const String& encode() const;

   /** Returns true if the URI is valid. */
   bool isValid() const { return m_bValid; }
      
   static void URLEncode( const String &source, String &target );
   static String URLEncode( const String &source )
   {
      String t;
      URLEncode( source, t );
      return t;
   }

   static void URLEncodePath( const String &source, String &target );
   static String URLEncodePath( const String &source )
   {
      String t;
      URLEncodePath( source, t );
      return t;
   }


   /**
     Decode an URI-URL encoded string.
     \param source the string to be decoded.
     \param target the target where to store the decoded string.
     \return true if the decoding was succesful, false otherwise.
   */
   static bool URLDecode( const String &source, String &target );
   static String URLDecode( const String &source )
   {
      String t;
      URLDecode( source, t );
      return t;
   }

   static unsigned char CharToHex( unsigned char ch )
   {
      return ch <= 9 ? '0' + ch : 'A' + (ch - 10);
   }

   static unsigned char HexToChar( unsigned char ch )
   {
      if ( ch >= '0' && ch <= '9' )
         return ch - '0';
      else if ( ch >= 'A' && ch <= 'F' )
         return ch - 'A'+ 10;
      else if ( ch >= 'a' && ch <= 'f' )
         return ch - 'a'+ 10;
      else
         return 0xFF;
   }
   
   //  A bit of parsing support.
   /** Character is a reserved delimiter under RFC3986 */
   inline static bool isResDelim( uint32 chr );

   /** Character is main section delimiter under RFC3986 */
   inline static bool isMainDelim( uint32 chr );

   /** Character is a general delimiter under RFC3986 */
   inline static bool isGenDelim( uint32 chr );

   /** Character is a subdelimiter under RFC4986 */
   inline static bool isSubDelim( uint32 chr );

   /** Unreserved characters under RFC3986 */
   inline static bool isUnreserved( uint32 chr );        
   
private:
   /** False if this URI is not valid. */
   bool m_bValid;

   /** URI scheme (e.g. http) */
   String m_scheme;   
   String m_authority;
   String m_path;
   String m_query;
   String m_fragment;
   

   /** The final normalized and encoded URI. */
   mutable String m_encoded;    
   
   bool internal_parse( const String &newUri );
};

//==================================================
// Inline implementation
//

inline bool URI::isResDelim( uint32 chr )
{
   return isGenDelim( chr ) || isSubDelim( chr );
}

inline bool URI::isMainDelim( uint32 chr )
{
   return (chr == ':') || (chr == '?') || (chr == '#') || (chr == '@');
}

inline bool URI::isGenDelim( uint32 chr )
{
   return (chr == ':') || (chr == '/') || (chr == '?') ||
          (chr == '#') || (chr == '[') || (chr == ']') ||
          (chr == '@');
}

inline bool URI::isSubDelim( uint32 chr )
{
   return (chr == '!')  || (chr == '$') || (chr == '&') ||
          (chr == '\'') || (chr == '(') || (chr == ')') ||
          (chr == '*')  || (chr == '+') | (chr == ',') ||
          (chr == ';')  || (chr == '=');
}

inline bool URI::isUnreserved( uint32 chr )
{
   return (chr == '-')  || (chr == '_') || (chr == '.') || (chr == '~') ||
          (chr >= 'a' && chr <= 'z') ||
          (chr >= 'A' && chr <= 'Z') ||
          (chr >= '0' && chr <= '9');
}

} // Falcon namespace

#endif
