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
#include <falcon/genericmap.h>

namespace Falcon
{

/** RFC 3986 - Uniform Resource Identifier.

   This class offer falcon engine and its users an interface to URI.
*/
class FALCON_DYN_CLASS URI: public BaseAlloc
{
   /** A copy of the original string, for diagniostics.*/
   String m_original;

   /** The final normalized and encoded URI. */
   mutable String m_encoded;

   /** False if this URI is not valid. */
   bool m_bValid;

   /** URI scheme (e.g. http) */
   String m_scheme;

   // Authority ----------------------
   /** User or user:password. */
   String m_userInfo;

   /** Host part of the URI. */
   String m_host;

   /** Port part of the URI. */
   String m_port;

   /** Path.
      Virtually divided into 4 elements:
      \code
      /[resource:]path/file.ext
      \endcode

      but calculus on path is done realtime.

      \note relative paths (not starting with /) cannot have a resource part.
   */
   Path m_path;

   /** Query string, recorded as-is. */
   mutable String m_query;

   /** Query part.
      Map of string->string.

      Will be used only if query parsing is explicitly requested.

      Many things using URI may not want this to be done, i.e. becasuse
      they want to parse the query on their own. Also, query form is free
      and may be a per-application format, even if the key=value& list is
      a quite eshtablished standard.
   */
   Map *m_queryMap;

   /** Iterator used for opaque traversal of query objects */
   MapIterator m_queryIter;

   /** Fragment. */
   String m_fragment;

   /** Encodes a prebuilt string which is then placed in the m_encoded field. */
   void encode( const String &u );

   /** Parses the query element. */
   bool internal_parseQuery( const String &str, uint32 pos, bool parseQuery , bool bDecode );

   /** Parses the fragment element. */
   bool internal_parseFragment( uint32 pos );

   bool internal_parse( const String &newUri, bool parseQuery, bool decode = true );

   friend class Path;
   
public:

   /** Empty constructor.
      Creates an empty URI. To be filled at a later time.
   */
   URI();

   /** Complete URI constructor.
      Decodes the uri and parses it in its fields.

      In case URI is not valid, isValid() will return false after construction.
   */
   URI( const String &suri );

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
      \param decode set to false to use the given URI as is.
      \param bMakeQueryMap if true, will create a string map with pre-parsed from query field, if present.
      \return true on success, false if the given string is not a valid URI.
   */
   bool parse( const String &newUri, bool parseQuery = false, bool decode = true );

   /** Parses the query field.
      Taken the query field of this class, it perform a RFC3986 scan for "&" separated
      keys and values pairs, each of which separated with a "=" sign.

      The result is set in the internal map of fields, that can then be inspected or
      changed keywise.

      This method overwrites existing keys with new ones, so it is not possible to
      use it to implement PHP-like URI arrays as in i.e.
      \code
         k[]=1&k[]=2
      \endcode

      \param decode true to automatically URL decode keys and values that will be stored in the map.
      \return true on success, false if the query field cannot be decoded.
   */
   bool parseQuery( bool decode = true );

   /** Changes query and the parses it field.
      This method calls in sequecnce the query() accessor and then the parseQuery() method.

      \param q a string that will be set as-is in the query field of this URI
      \param decode true to automatically URL decode keys and values that will be stored in the map.
      \return true on success, false if the query field cannot be decoded.
   */
   bool parseQuery( const String &q, bool decode = true )
   {
      query( q );
      return parseQuery( decode );
   }

   /** Sets the query field of this URI.
      The query field is set as-is. This destroys previously created maps of
      keys values that should be used as query field generators.

      The parameter should be URL encoded before being set into this method,
      or the \b encode parameter may be used to have this method to perform
      URL encoding.

      \param q the query to be set.
      \param encode if true, q is considered a plain string still to be encoded.
   */
   void query( const String &q, bool encode = false );

   /** Returns previously set query.
      This method returns a previously set query field as-is.

      The content of the key-value map of this URI object, if any, is ignored.

      \note To make a query field out of a query-map, use the makeQuery() method.
   */
   const String &query() const { return m_query; }

   /** Synthetizes a query field out of the key-values stored in this URI object.

      This call clears the content of the query field and changes it with an
      RFC3986 encoded key-value pair list in the format
      \code
         k1=v1&k2=v2&...&kn=vn
      \endcode

   \return The synthetized string.
   */
   const String &makeQuery() const;

   /** Returns the current URI.
      This method eventually builds a new URI from the internally parsed data
      and returns it.

      If a set of key-value pairs has been set in this URI, it is used to
      synthetize a query field using makeQuery() method. If this is not desired,
      i.e. because already done, or because the query field has been set
      separately, the \b synthQuery parameter may be set to false, and the content
      of the query field will be used instead.
   */
   const String &get( bool synthQuery = true ) const;

   //  A bit of parsing support.
   /** Character is a reserved delimiter under RFC3986 */
   inline static bool isResDelim( uint32 chr );

   /** Character is main section delimiter under RFC3986 */
   inline static bool isMainDelim( uint32 chr );

   /** Character is a general delimiter under RFC3986 */
   inline static bool isGenDelim( uint32 chr );

   /** Character is a subdelimiter under RFC4986 */
   inline static bool isSubDelim( uint32 chr );

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

   /** Returns current scheme. */
   const String &scheme() const { return m_scheme; }

   /** Sets a different scheme for this URI.
      This will invalidate current URI until next uri() is called.
   */
   void scheme( const String &s );

   /** Returns current userInfo. */
   const String &userInfo() const { return m_userInfo; }

   /** Sets a different userInfo for this URI.
      This will invalidate current URI until next uri() is called.
   */
   void userInfo( const String &s );

   /** Returns current host. */
   const String &host() const { return m_host; }

   /** Sets a different host for this URI.
      This will invalidate current URI until next uri() is called.
   */
   void host( const String &h );

   /** Returns current port. */
   const String &port() const { return m_port; }

   /** Sets a different port for this URI.
      This will invalidate current URI until next uri() is called.
   */
   void port( const String &h );

   /** Returns current path. */
   const String &path() const { return m_path.get(); }

   /** Returns current path as a path class. */
   const Path &pathElement() const { return m_path; }
   
   /** Returns current path as a path class. */
   Path &pathElement() { return m_path; }

   /** Sets a different path for this URI.
      This will invalidate current URI until next uri() is called.
   */
   void path( const String &p );

   /** Sets a different path for this URI.
      This will invalidate current URI until next uri() is called.
   */
   void path( const Path &p );

   /** Returns true if the query part has this field. */
   bool hasField( const String &f ) const ;

   /** Returns the required value, if it exists. */
   bool getField( const String &key, String &value ) const;

   /** Sets a given query field.
      As a convention, if the string contains only a single 0 character (NUL), the
      final result won't include = in the query part, while an empty string will
      result in a query string containing only a keu and a "=" followed by nothing.
      Strings longer than 1 element are not interpreted this way, so a nul followed
      by some data would be rendere as "%00" followed by the rest of the value.
   */
   void setField( const String &key, const String &value );

   /** Removes a query field. */
   bool removeField( const String &key );

   /** Enumerates the query fields - gets the first field.
      Returns true if there is a first field in the query.
      \note The query element must have been previously parsed, or fields must have
      been explicitly inserted.
      \param key a string where the key of the first field will be placed
      \param value a string where the value of the first field will be placed
         (can be an empty string).

      \return true if there is a first field.
   */
   bool firstField( String &key, String &value );

   /** Enumerates the query fields - gets the next field.
      Returns true if there is a next field.
      \note The query element must have been previously parsed, or fields must have
      been explicitly inserted.

      \param key a string where the key of the first field will be placed
      \param value a string where the value of the first field will be placed
         (can be an empty string).
   */
   bool nextField( String &key, String &value );

   /** Enumerates the query fields - counts the fields.
      If the query has fields, or if fields have been explicitly set throug
      setField() method, returns the count of fields stored in this URI.

      \note The query element must have been previously parsed, or fields must have
      been explicitly inserted.

      \return count of fields in this query, 0 for none.
   */
   uint32 fieldCount();

   /** Returns the fragment part. */
   const String &fragment() const { return m_fragment; }

   /** Sets the fragment part. */
   void fragment( const String &s );

   /** Clears the content of this URI */
   void clear();

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
};

//==================================================
// Inline implementation
//

inline bool URI::isResDelim( uint32 chr )
{
   return isGenDelim( chr ) | isSubDelim( chr );
}

inline bool URI::isMainDelim( uint32 chr )
{
   return (chr == ':') | (chr == '?') | (chr == '#') | (chr == '@');
}

inline bool URI::isGenDelim( uint32 chr )
{
   return (chr == ':') | (chr == '/') | (chr == '?') |
          (chr == '#') | (chr == '[') | (chr == ']') |
          (chr == '@');
}

inline bool URI::isSubDelim( uint32 chr )
{
   return (chr == '!')  | (chr == '$') | (chr == '&') |
          (chr == '\'') | (chr == '(') | (chr == ')') |
          (chr == '*')  | (chr == '+') | (chr == ',') |
          (chr == ';')  | (chr == '=');
}

} // Falcon namespace

#endif
