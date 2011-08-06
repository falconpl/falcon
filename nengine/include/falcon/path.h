/*
   FALCON - The Falcon Programming Language.
   FILE: path.h

   RFC 3986 compliant file path definition.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 27 Feb 2008 22:03:00 +0100

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_PATH_H
#define FALCON_PATH_H

#include <falcon/setup.h>
#include <falcon/string.h>


namespace Falcon
{
class URI;

/** Falcon path representation.

   This class is actually a string wrapper which parses the path and builds it as necessary.

   With respect to a string, 0 overhead is required.

   However, notice that this class cannot be used as a garbage string, and must be wrapped
   into a UserData to be part of a falcon object.

   Path must be provided in Falcon format (RFC 3986): path elements must be separated by forward
   slashes and resource identifiers must be preceded by a single "/"; in example:
   \code
      /C:/falcon/file.fal
   \endcode
   With a resource identifier, the first "/" is optional when setting the path, but the
   internal representation will be normalized so that it is present.

   Methods to transfrorm this representation to and from MS-Windows path are provided.

   The path is not internally checked, by this class, so any string may be set,
   but it may get checked i.e. when insernted in a URI.
*/

class FALCON_DYN_CLASS Path
{
public:

   /** Empty constructor. */
   Path();

   /** Path constructor from strings. */
   Path( const String &path ):
      m_bValid( true )
   {
      parse( path );
   }

   /** Copy constructor.
      Copies the other path as-is.
   */
   Path( const Path &other ):
      m_bValid( true )
   {
      copy( other );
   }

   /** Copy another path.
      Copies the other path as-is.
   */
   void copy( const Path &other );

   /** Set a path from RFC 3986 format. 
    The parsing automatically detects windows format and.
    */
   bool parse( const String &p );

   /** Retrurn the path in RFC 3986 format. */
   const String &encode() const
   {
      if( m_encoded.size() > 0)
      {
         return m_encoded;
      }
      encode_internal();
      return m_encoded;
   }

   bool isValid() const { return m_bValid; }
   
   void clear();
   
   //=================================================================
   // Setting and getting raw elements.
   //
   
   /** Get the location part (path to file) in RFC3986 format.
   */
   const String& location() const { return m_location; }
   
   /** Sets the location part in RFC3986 format. 
    \param loc the new location.
    \return false If the location is malformed (cannot contain ':').
    
    The location is the part of the path that indicates the directory
    (on a certain unit) where a file resides.
    
    Backslashes in the location are automatically converted into forward
    slashes.
    
    */
   bool location( const String &loc );

 
   /** Get the filename part.
      This returns the file and extension parts separated by a '.'.
   */
   const String& fileext() const { encode(); return m_filename; }
   
   /** Sets the filename and the extension.
    \param fn The file and extension.
    \return true if the format is correct, false if the file contains slashes,
    colons or semicolons.
    This returns the file and extension parts separated by a '.'.
    */
   bool fileext( const String& fn );
    

   /** Get the file part alone (without extension). */
   const String& file() const { return m_file; }

   
   /** Sets the file part (without extension). 
     \param value the new file part.
    \return true if the part can be changed, false if it contains
    an invalid character.
    */
   bool file( const String& value );
  
   /** Get the extension part. */
   const String& ext() const { return m_extension; }
   
   /** Changes the extension part. 
     \param value the new extension part.
    \return true if the extension can be changed, false if it contains
    an invalid character.
    */
   bool ext( const String& value );
       
   /** Get the resource specificator part (path to file) in RFC3986 format.
   */
   const String& resource() const { return m_device; }
   
   /** Sets the resource specificator part in RFC3986 format. 
    \param dev the new resource.
    
    The device is the part of the path that indicates a unit,
    a device or a generic resource indicator (e.g. C for "C:\\" on
    windows).
    
    The dev parameter cannot have any separator (colon, slashes, semicolon,
    dots).
    */
   bool resource( const String &dev );
      

   //=========================================================
   // Andvanced utilities.
   //

   /** Returns a path in MS-Windows format. 
    \return The path stored in this object in MS-Windows path format.
    \note This method returns a temporary string. Use with care.
    */
   
   String getWinFormat() const { String fmt; getWinFormat( fmt ); return fmt; }
   
   /** Stores this path in windows format in a given string. 
    \param str Where to store the path in windows format.
    */
   void getWinFormat( String &str ) const;


   /** Gets the location part, eventually including the resource specificator if present. 
    \return 
    */
   String getFullLocation() const { String fmt; getFullLocation( fmt ); return fmt; }
   
   /** Gets the location part, eventually including the resource specificator if present. 
    \param str Where to store the full location (device + path).
    */
   bool getFullLocation( String &str ) const;


   /** Get the location part (path to file) in MS-Windows format. */
   String getWinLocation() const { String fmt; getWinLocation( fmt ); return fmt; }

   /** Stores the location part in a given string in MS-Windows format.
      If the path has not a location part, the string is also cleaned.
      \param str the string where to store the location part.
      \return true if the path has a location part.
   */
   bool getWinLocation( String &str ) const;

   /** Get Windows disk specificator + location (windows absolute path).

      \param str the string where to store the location part.
      \return true if the path has a disk or location part.
   */
   bool getFullWinLocation( String &str ) const;

   /** returns the disk specificator + location (windows absolute path) */
   String getFullWinLocation() const { String fmt; getFullWinLocation( fmt ); return fmt; } 

   /** Sets both the resource and the location in one step. 
      If the parameter is empty, both location and resource are cleared.
      
      If the parameter is just a resource specificator (i.e. "C:" or "/C:"),
      then the location is cleared.
      
      If it's just a location, then the resource is cleared.

      May return false if the parsing of the new content fails.
      \param floc Full location.
      \return true if correctly parsed.
   */
   bool setFullLocation( const String &floc );
   
   /** Returns true if this path is an absolute path. */
   bool isAbsolute() const;

   /** Returns true if this path defines a location without a file */
   bool isLocation() const;


   /** Splits the path into its constituents.
      \param loc a string where the location will be placed.
      \param name a string where the filename in this path will be placed.
      \param ext a string where the file extension will be placed.
   */
   void split( String &loc, String &name, String &ext );

   /** Splits the path into its constituents.
      This version would eventually put the resource part in the first parameter.
      \param res a string where the resource locator will be placed.
      \param loc a string where the location will be placed.
      \param name a string where the filename in this path will be placed.
      \param ext a string where the file extension will be placed.
   */
   void split( String &res, String &loc, String &name, String &ext );

   /** Splits the path into its constituents.
      This version will convert the output loc parameter in MS-Windows path format
         (backslashes).
      \param res a string where the resource locator will be placed.
      \param loc a string where the location will be placed.
      \param name a string where the filename in this path will be placed.
      \param ext a string where the file extension will be placed.
   */
   void splitWinFormat( String &res, String &loc, String &name, String &ext );

   /** Joins a path divided into its constituents into this path.
      Using this version it is not possible to set a resource locator (i.e. a disk unit).

      \param loc the path location of the file.
      \param name the filename.
      \param ext the file extension.
   */
   void join( const String &loc, const String &name, const String &ext );

   /** Joins a path divided into its constituents into this path.
      \param res the resource locator (i.e. disk unit)
      \param loc the path location of the file.
      \param name the filename.
      \param ext the file extension.
   */
   void join( const String &res, const String &loc, const String &name, const String &ext );

   /** Add a path element at the end of a path.
    \param npath The new part of the path to add to the location.
    \return true on success, false if the path is malformed.
    
      This extens the path adding some path element after the currently
      existing location portion. Leading "/" in npath, or trailing "/" in this
      path are ignored, and a traling "/" is forcefully added if there is a file
      element. In example, adding p1/p2 or /p1/p2 through this method:

      /C:file.txt  => /C:/p1/p2/file.txt
      /path/ => /path/p1/p2
      /path/file.txt => /path/p1/p2/file.txt
   */
   bool extendLocation( const String &npath );

   Path & operator =( const Path &other ) { copy( other ); return *this; }
   bool operator ==( const Path &other ) const { encode(); other.encode(); return other.m_encoded == m_encoded; }
   bool operator !=( const Path &other ) const { encode(); other.encode(); return other.m_encoded != m_encoded; }
   bool operator <( const Path &other ) const { encode(); other.encode(); return m_encoded < other.m_encoded; }
   bool operator >( const Path &other ) const { encode(); other.encode(); return m_encoded < other.m_encoded; }
   

   /** Converts an arbitrary MS-Windows Path to URI path.
    An URI valid path starts either with a filename or with a "/"
    if absolute. It can't start with a MS-Windows disk or unit
    specifier as c:\\. Also, it can't contain backslashes.

    This static function transforms the path parameter in place
    so that it is changed into a valid URI path. For example:
    \code
       path\\to\\file.txt  => path/to/file.txt
       \\path\\to\\file.txt => /path/to/file.txt
       c:\\path\\to\\file.txt => /c:/path/to/file.txt
    \endcode
    @param path the path to be converted on place
   */
   static void winToUri( String &path );

   /** Converts an arbitrary URI path into a MS-Windows path.
    An URI valid path starts either with a filename or with a "/"
    if absolute. It can't start with a MS-Windows disk or unit
    specifier as c:\\. Also, it can't contain backslashes.

    This static function transforms the path parameter in place
    so that it is changed into a valid MSWindows path. For example:
    \code
        path/to/file.txt => path\\to\\file.txt
        /path/to/file.txt => \\path\\to\\file.txt
        /c:/path/to/file.txt = c:\\path\\to\\file.txt
    \endcode

      @note This function won't complain nor emit any warning
      if path is not a valid URI path in the first place.
      It's pretty dumb, but efficient.

    @param path the path to be converted on place
    @see getWinFormat
   */
   static void uriToWin( String &path );
   
private:
   mutable String m_encoded;
   mutable String m_filename;
   
   String m_device;
   String m_location;
   String m_file;
   String m_extension;

   bool m_bValid;

   void encode_internal() const;
};

}

#endif
