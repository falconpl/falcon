/*
   FALCON - The Falcon Programming Language.
   FILE: confparser_ext.cpp

   Falcon VM interface to confparser module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2006-05-09 15:50

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Falcon VM interface to configuration parser module.
*/


#include <falcon/fassert.h>
#include <falcon/vm.h>
#include <falcon/string.h>
#include <falcon/carray.h>
#include <falcon/lineardict.h>
#include <falcon/stream.h>
#include <falcon/memory.h>

#include "confparser_ext.h"
#include "confparser_mod.h"
#include "confparser_st.h"

/*#
   @beginmodule feather_configparser
*/

namespace Falcon {
namespace Ext {

// ==============================================
// Class ConfParser
// ==============================================

/*#
   @class ConfParser
   @brief Interface to configuration files.
   @optparam filename The name of the ini file to be parsed, if it exists.
   @optparam encoding An optional non default encoding which is used for that file.

   The constructor of this class allows to set up a filename for the
   configuration file that is going to be read and/or written. If the name is not
   given, @a ConfParser.read and ConfParser.write methods will require a valid Falcon
   Stream, otherwise, if the stream is not provided, the given file will be opened
   or written as needed.

   @section confparser_key_manage Adding, setting or removing keys

   The most direct way to add a new key in an ini configuration file is to use the
   @a ConfParser.add method.

   If the key had not any previous value, a new key is added. If a specified
   section was not previously present it is added.

   If one or more entries with the same key was already present, the entry will be
   physically placed as the last one, and if queried at a later moment, the value
   will be returned as the last value of the value arrays associated with the given
   key.

   The value parameter may also be an array of strings, in which case all the
   values contained in the array will be added, one after another as specified for
   the single value case. If the caller wants to be sure that only the values in
   the given value or value array are set, it should call the @a ConfParser.remove method
   before calling @a ConfParser.add.

   To set a single value, eventually getting rid of other previously existing
   values, use the set method: @a ConfParser.set, which sets a key/value pair in the main section,
   or if section parameter is given and not nil, in the specified section.
   Using this method, any previous value or value set associated with the given key
   is removed. If the key had not any previous value, a new key is added. If a
   specified section was not previously present, it is added. If the given key was
   already present in a parsed configuration file, it's position and the comments
   that were eventually associated with the key are left unchanged.

   To remove completely a key, use the @a ConfParser.remove method. To remove
   completely a section, use the removeSection( section ) method. This method can't
   be used to remove the main section; in fact, even if empty, that section always
   exists. To clear every key in that, use the clearMain() method.

   @section confparser_categorized_keys Categorized keys

   Categories are separated from the keys by dots "."; a complete categorized key
   may contain any amount of dots, or in other words, the categories can have an
   arbitrary depth.

   The getCategoryKeys method returns a list of all the keys belonging to a certain
   category. Categories are separated from the keys by dots "."; a complete
   categorized key may contain any amount of dots, or in other words, the
   categories can have an arbitrary depth.

   The category (first) parameter of this method may indicate the first level
   category only, or it can be arbitrarily deep. Only the keys in deeper categories
   will be returned.

   For example; if the configuration file contains the following entries:

   @code
   Key.cat1 = 1
   Key.cat1.k1 = 101
   Key.cat1.k2 = 102
   Key.cat1.k3 = 103
   Key.cat1.subcat1.k1 = 105
   Key.cat1.subcat1.k2 = 106
   @endcode

   if the category parameter is set to "Key", all the entries will be returned. If
   it's set to "cat1", the first entry won't be returned, as it's considered a key
   cat1 in category Key. If category is set to "key.cat1.subcat1", the last two
   entries will be returned.

   The strings in the returned array represent the complete key name, including the
   complete categorization. In this way it is directly possible to retrieve the
   value of a given key, or to alter their values, by simply iterating on the
   returned array, like in the following example:

   @code
   category = "Key.cat1.subcat1"
   trimming = [len(category)+1:]

   keys = parser.getCategoryKeys( category )
   printl( "Keys in category ", category, ": " )
   for key in keys
      printl( key[ trimming ], "=", parser.get( key ) )
   end
   @endcode

   The result will be:

   @code
   Keys in category Key.cat1.subcat1:
   k1=105
   k2=106
   @endcode

   If the category cannot be found, or if it doesn't contain any entry, or if a
   section parameter is provided but the given section cannot be found, this method
   returns an empty array. It is necessary to ascertain that the requested values
   are present (of if not, that their missing actually means that the category is
   "empty") by other means.

   Other than just enumerating categorized keys, that can then be read with the
   ordinary get() or getOne() methods, a whole category tree can be imported with
   the method getCategory().

   For example, consider the same configuration structure we have used before. If
   the category parameter  of getCategory() is set to "Key", all the entries will
   be returned. If it's set to "cat1", the first entry won't be returned, as it's
   consider a key cat1 in category Key. If category is set to "key.cat1.subcat1",
   the last two entries will be returned.

   The strings returned in the dictionary keys are the complete key, including the
   category part. It is possible to obtain a dictionary where the keys are already
   stripped of their category part by adding an asterisk at the end of the first
   parameter.

   For example:

   @code
   category = "Key.cat1"
   valueDict = parser.getCategory( category+"*" )

   printl( "Keys in category ", category, ": " )
   for key, value in valueDict
      printl( key, "=", value )
   end
   @endcode

   The result will be:

   @code
   Keys in category Key.cat1:
   k1=101
   k2=102
   k3=103
   subcat1.k1=105
   subcat1.k2=106
   @endcode

   If a key has multiple values, it's value element will be set to an array
   containing all the values.
*/

FALCON_FUNC  ConfParser_init( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item *i_fname = vm->param(0);
   Item *i_encoding = vm->param(1);

   if ( (i_fname != 0 && ! i_fname->isString()) || ( i_encoding != 0 && ! i_encoding->isString() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S, [S]" ) );
      return;
   }

   String fname;
   String encoding;

   if ( i_fname != 0 )
      fname = *i_fname->asString();

   if ( i_encoding != 0 )
      encoding = *i_encoding->asString();

   ConfigFile *cfile = new ConfigFile( fname, encoding );
   self->setUserData( cfile );
}


/*#
   @method read ConfParser
   @brief Read the ini file.
   @optparam stream An optional input stream from where to read the file.
   @raise IoError on read error.

   Parses a configuration file and prepares the object data that may be retrieved
   with other methods. The @b read method may be provided with an opened and
   readable Falcon stream. If not, the file name provided to the ConfParser
   constructor will be opened and read. In case the name has not been given in the
   constructor, the method raises an error. The method may also raise ParseError,
   IoError or ParamError, with the "message" field set to a relevant explanation.
*/

FALCON_FUNC  ConfParser_read( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   Item *i_stream = vm->param(0);

   bool bRes;

   if( i_stream == 0 )
   {
      vm->idle();
      bRes = cfile->load();
      vm->unidle();
   }
   else {
      bool bValid = false;
      if ( i_stream->isObject() )
      {
         CoreObject *streamObj = i_stream->asObject();
         if ( streamObj->derivedFrom( "Stream" ) )
         {
            Stream *base = (Stream *) streamObj->getUserData();
            bRes = cfile->load( base );
            bValid = true;
         }
      }

      if ( ! bValid )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "Stream" ) );
         return;
      }
   }

   if ( ! bRes )
   {
      // is this an I/O or a parsing error?
      if ( cfile->fsError() != 0 )
      {
         throw new IoError( ErrorParam( e_loaderror, __LINE__ ).
            sysError( cfile->fsError() ).
            extra( cfile->errorMessage() ) );
      }
      else {
         String msg = cfile->errorMessage() + " at ";
         msg.writeNumber( (int64) cfile->errorLine() );
         self->setProperty( "error", cfile->errorMessage() );
         self->setProperty( "errorLine", (int64) cfile->errorLine() );
         throw new ParseError( ErrorParam( FALCP_ERR_INVFORMAT, __LINE__ )
            .desc( FAL_STR(cp_msg_invformat) )
            .extra( msg ) );
      }
   }

}

/*#
   @method write ConfParser
   @brief Write the INI file.
   @optparam stream An optional output stream on which to write the configuration file.
   @raise IoError on write error.

   Writes the content of a modified or entirely generated configuration file on the
   given stream, that must be a valid Falcon stream opened for output. If a stream
   is not given, then the file name provided to the ConfParser constructor is
   opened for writing. In case the name has not been given in the constructor, the
   method raises an error.

*/
FALCON_FUNC  ConfParser_write( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();

   Item *i_stream = vm->param(0);

   bool bRes;

   if( i_stream == 0 )
   {
      bRes = cfile->save();
   }
   else {
      bool bValid = false;
      if ( i_stream->isObject() )
      {
         CoreObject *streamObj = i_stream->asObject();
         if ( streamObj->derivedFrom( "Stream" ) )
         {
            Stream *base = (Stream *) streamObj->getUserData();
            bRes = cfile->save( base );
            bValid = true;
         }
      }

      if ( ! bValid )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "Stream" ) );
         return;
      }
   }

   if ( ! bRes )
   {
      // is this a file error?
      if ( cfile->fsError() )
      {
         throw new IoError( ErrorParam( e_file_output, __LINE__ ).
            sysError( cfile->fsError() ).
            extra( cfile->errorMessage() ) );
      }
      else
      {
         // no -- it's a configuration file.d
         self->setProperty( "error", cfile->errorMessage() );
         self->setProperty( "errorLine", (int64) cfile->errorLine() );
         throw new ParseError( ErrorParam( FALCP_ERR_STORE, __LINE__ ).
            desc( FAL_STR(cp_msg_errstore)  ).extra( cfile->errorMessage() ) );
      }
   }
}

/*#
   @method get ConfParser
   @brief Retreives the value associated with a key.
   @param key The key of which the value is to be read.
   @optparam section If provided, the section where the key is found.
   @return The value (or values) of associated to the key, or nil if not found.

   The method retrieves the value associated with a given key. If section parameter
   is not provided, or if it's nil, the key is searched in the main section, else
   it is searched in the given section.

   If the section does not exist, or if the key is not present in the given
   section, the method returns nil. If the key exist but has no value associated
   with it, an empty string is returned. If there is only one instance of the key,
   a single string containing the value is returned. If multiple entries for the
   given key are found, all the values are returned as strings in an array.
   The caller should verify the if the returned value is a string or an array using
   typeOf() function. Alternatively, it is possible to use @a ConfParser.getOne to be sure to
   retrieve only strings.

   Categorized keys can be retrieved with this method by providing their full name.
*/
FALCON_FUNC  ConfParser_get( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   Item *i_key = vm->param(0);
   Item *i_section = vm->param(1);

   if ( i_key == 0 || ! i_key->isString() ||
        ( i_section != 0 && ! i_section->isString() && ! i_section->isNil() )
      )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ) );
   }

   String value;

   if ( i_section != 0 && ! i_section->isNil() )
   {
      if ( ! cfile->getValue( *i_section->asString(), *i_key->asString(), value ) )
      {
         vm->retnil();
         return;
      }
   }
   else {
      if ( ! cfile->getValue( *i_key->asString(), value ) )
      {
         vm->retnil();
         return;
      }
   }

   // we have at least one value. but do we have more?
   String value1;
   if ( cfile->getNextValue( value1 ) )
   {
      CoreArray *array = new CoreArray( 5 );
      array->append( new CoreString( value ) );
      array->append( new CoreString( value1 ) );

      while( cfile->getNextValue( value1 ) )
         array->append( new CoreString( value1 ) );

      vm->retval( array );
   }
   else {
      vm->retval( value );
   }
}

/*#
   @method getOne ConfParser
   @brief Retreives the value associated with a key.
   @param key The key of which the value is to be read.
   @optparam section If provided, the section where the key is found.
   @return The value (or values) of associated to the key, or nil if not found.

   This method is equivalent to the @a ConfParser.get method, except for the fact that if more
   than one value has been given for the determined key in the configuration file,
   only the last one among them is returned.
*/
FALCON_FUNC  ConfParser_getOne( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   Item *i_key = vm->param(0);
   Item *i_section = vm->param(1);

   if ( i_key == 0 || ! i_key->isString() ||
        ( i_section != 0 && ! i_section->isString() && ! i_section->isNil() )
      )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ) );
   }

   String value;

   if ( i_section != 0 && ! i_section->isNil() )
   {
      if ( ! cfile->getValue( *i_section->asString(), *i_key->asString(), value ) )
      {
         vm->retnil();
         return;
      }
   }
   else {
      if ( ! cfile->getValue( *i_key->asString(), value ) )
      {
         vm->retnil();
         return;
      }
   }

   vm->retval( value );
}

/*#
   @method getMultiple ConfParser
   @brief Retreives the value associated with a key.
   @param key The key of which the value is to be read.
   @optparam section If provided, the section where the key is found.
   @return All the values of associated to the key, or nil if not found.

   This method is equivalent to @a ConfParser.get method, except for the fact that an array of
   values is always returned even if only one key is found. If there is no entry in
   the configuration file coresponding to the given key, nil is returned.
*/
FALCON_FUNC  ConfParser_getMultiple( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   Item *i_key = vm->param(0);
   Item *i_section = vm->param(1);

   if ( i_key == 0 || ! i_key->isString() ||
        ( i_section != 0 && ! i_section->isString() && ! i_section->isNil() )
      )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ) );
   }

   String value;
   if ( i_section != 0 && ! i_section->isNil() )
   {
      if ( ! cfile->getValue( *i_section->asString(), *i_key->asString(), value ) )
      {
         vm->retnil();
         return;
      }
   }
   else {
      if ( ! cfile->getValue( *i_key->asString(), value ) )
      {
         vm->retnil();
         return;
      }
   }

   CoreArray *array = new CoreArray( 5 );
   array->append( new CoreString( value ) );

   String value1;
   while( cfile->getNextValue( value1 ) )
      array->append( new CoreString( value1 ) );

   vm->retval( array );
}


/*#
   @method getSections ConfParser
   @brief Enumerates the sections that are declared in the file managed by this object.
   @return All the values of associated to the key, or nil if not found.

   If the object doesn't declare any section, the method returns an empty array.
*/
FALCON_FUNC  ConfParser_getSections( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();

   String section;
   CoreArray *ret = new CoreArray;

   if( cfile->getFirstSection( section ) )
   {
      ret->append( new CoreString( section ) );
      while( cfile->getNextSection( section ) )
         ret->append( new CoreString( section ) );
   }

   vm->retval( ret );
}

/*#
   @method getKeys ConfParser
   @brief Retreives the value associated with a key.
   @optparam section An optional section on which to operate.
   @return All the keys listed in the given section, or in the main part.

   This method returns an array of strings containing all the keys in the main
   section, or if a section parameter is given and not nil, it returns all the keys
   in the given section.

   If the given section exists but it doesn't contain any key, an empty array is
   returned. If the section doesn't exist, the method returns nil.
*/
FALCON_FUNC  ConfParser_getKeys( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   Item *i_section = vm->param( 0 );

   if ( i_section != 0 && ! i_section->isString() && ! i_section->isNil() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ) );
      return;
   }

   String key;
   CoreArray *ret = new CoreArray;
   bool next;

   if ( i_section != 0 && ! i_section->isNil() ) {
      next = cfile->getFirstKey( *i_section->asString(), "", key );
   }
   else {
      next = cfile->getFirstKey( "", key );
   }

   while ( next )
   {
      ret->append( new CoreString( key ) );
      next = cfile->getNextKey( key );
   }

   vm->retval( ret );
}

/*#
   @method getCategoryKeys ConfParser
   @brief Get the keys filed under a given category.
   @param category The category of which the key list is required
   @optparam section If provided, the section where the category is defined.
   @return All the keys listed in the given category.

   This method returns a list of all the keys belonging to a certain category.

   See the "Categorized keys" section in @a ConfParser.
*/
FALCON_FUNC  ConfParser_getCategoryKeys( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   Item *i_keyMask = vm->param( 0 );
   Item *i_section = vm->param( 1 );

   if ( i_keyMask == 0 || ! i_keyMask->isString() ||
        ( i_section != 0 && ! i_section->isString() && ! i_section->isNil() )
      )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ) );
      return;
   }

   String key;
   CoreArray *ret = new CoreArray;
   bool next;

   if ( i_section != 0 && ! i_section->isNil() ) {
      next = cfile->getFirstKey( *i_section->asString(), *i_keyMask->asString(), key );
   }
   else {
      next = cfile->getFirstKey( *i_keyMask->asString(), key );
   }

   while ( next )
   {
      ret->append( new CoreString( String( key, i_keyMask->asString()->length() + 1 ) ) );
      next = cfile->getNextKey( key );
   }

   vm->retval( ret );
}


/*#
   @method getCategory ConfParser
   @brief Retreives keys and values given under a certain category.
   @param category The category of which the values are required
   @optparam section If provided, the section where the category is defined.
   @return A dictionary containing a pair of key-values in the given category.

   This method returns a dictionary of key-value pairs containing all the keys
   and values in a certain category.

   See the "Categorized keys" section in @a ConfParser.
*/
FALCON_FUNC  ConfParser_getCategory( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   Item *i_keyMask = vm->param( 0 );
   Item *i_section = vm->param( 1 );

   if ( i_keyMask == 0 || ! i_keyMask->isString() ||
        ( i_section != 0 && ! i_section->isString() && ! i_section->isNil() )
      )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ) );
   }

   if ( i_section != 0 && i_section->isNil() )
      i_section = 0;

   String key, keymask;
   LinearDict *ret = new LinearDict();
   LinearDict *current = ret;
   bool next;

   bool stripNames;
   keymask = *i_keyMask->asString();
   if ( keymask.length() > 0 && keymask.getCharAt(keymask.length() - 1) == '*' )
   {
      stripNames = true;
      keymask.size( keymask.size() - keymask.manipulator()->charSize() );
   }
   else
      stripNames = false;

   if ( keymask.length() > 0 && keymask.getCharAt(keymask.length() - 1) == '.' )
      keymask.size( keymask.size() - keymask.manipulator()->charSize() );

   if ( i_section != 0  ) {
      next = cfile->getFirstKey( *i_section->asString(), keymask, key );
   }
   else {
      next = cfile->getFirstKey( keymask, key );
   }

   while( next )
   {
      String value;

      // seeking a value won't alter key iterators.
      if( i_section != 0  )
         cfile->getValue( *i_section->asString(), key, value );
      else
         cfile->getValue( key, value );


      // we have at least one value. but do we have more?
      String value1;
      if ( cfile->getNextValue( value1 ) )
      {
         CoreArray *array = new CoreArray( 5 );
         array->append( new CoreString( value ) );
         array->append( new CoreString( value1 ) );

         while( cfile->getNextValue( value1 ) )
            array->append( new CoreString( value1 ) );

         // we have used KEY; now what we want to save is just the non-category
         if ( stripNames )
            current->put( new CoreString( key, keymask.length() + 1 ), array );
         else
            current->put( new CoreString( key), array );
      }
      else {
          if ( stripNames )
            current->put( new CoreString( key, keymask.length() + 1 ), new CoreString( value ) );
         else
            current->put(  new CoreString( key) , new CoreString( value ) );
      }

      next = cfile->getNextKey( key );
   }

   vm->retval( new CoreDict(ret) );
}


/*#
   @method getDictionary ConfParser
   @brief Retreives keys and values given under a certain category.
   @optparam section If given, the section from which to extract the dictionary.
   @return A dictionary containing a pair of key-values in the given section.

   This method retrieves all the pairs of key and values in the main section, or if
   a non-nil section parameter is provided, from the given section. If the
   requested section cannot be found, or if it doesn't contain any entry, an empty
   dictionary is returned. If a key has multiple values, its element is set to an
   array containing all the values.
*/
FALCON_FUNC  ConfParser_getDictionary( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   Item *i_section = vm->param( 0 );

   if ( i_section != 0 && ! i_section->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ) );
   }

   String key;
   LinearDict *ret = new LinearDict();
   LinearDict *current = ret;
   bool next;

   if ( i_section != 0 ) {
      next = cfile->getFirstKey( *i_section->asString(), "", key );
   }
   else {
      next = cfile->getFirstKey( "", key );
   }

   while( next )
   {
      String value;

      // seeking a value won't alter key iterators.
      if( i_section != 0 )
         cfile->getValue( *i_section->asString(), key, value );
      else
         cfile->getValue( key, value );

      // we have at least one value. but do we have more?
      String value1;
      if ( cfile->getNextValue( value1 ) )
      {
         CoreArray *array = new CoreArray( 5 );
         array->append( new CoreString( value ) );
         array->append( new CoreString( value1 ) );

         while( cfile->getNextValue( value1 ) )
            array->append( new CoreString( value1 ) );

         current->put( new CoreString( key ), array );
      }
      else {
         current->put( new CoreString( key ), new CoreString( value ) );
      }

      next = cfile->getNextKey( key );
   }

   vm->retval( new CoreDict(ret) );
}

/*#
   @method add ConfParser
   @brief Adds a key/value pairs to the INI file.
   @param key The key to which add the given value.
   @param value The value, or value array, to be added.
   @optparam section If provided, the section where to add the entry

   This function adds a key/value pair to the main section, or if section parameter
   is given and not @b nil, to the specified section.

   If the key is already present, a multiple value is set.
*/
FALCON_FUNC  ConfParser_add( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   Item *i_key = vm->param(0);
   Item *i_value = vm->param(1);
   Item *i_section = vm->param(2); // actually, if valorized, key and value are param 1 and 2.

   if ( i_key == 0 || ! i_key->isString() || i_value == 0 ||
        ( i_section != 0 && ! i_section->isString() && ! i_section->isNil() )
      )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S, S" ) );
   }

   String *value;
   bool delValue;
   if( i_value->isString() )
   {
      delValue = false;
      value = i_value->asString();
   }
   else {
      value = new CoreString;
      delValue = true;
      vm->itemToString( *value, i_value );
   }

   if( i_section == 0 || i_section->isNil() )
      cfile->addValue( *i_key->asString(), *value );
   else
      cfile->addValue( *i_section->asString(), *i_key->asString(), *value );

   if ( delValue )
      delete value;
}

/*#
   @method set ConfParser
   @brief Sets the value of a certain key key.
   @param key The key to which add the given value.
   @param value The value, or value array, to be added.
   @optparam section If provided, the section where to add the entry

   Sets a key/value pair in the main section, or if section parameter is
   given and not nil, in the specified section.
*/
FALCON_FUNC  ConfParser_set( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   Item *i_key = vm->param(0);
   Item *i_value = vm->param(1);
   Item *i_section = vm->param(2); // actually, if valorized, key and value are param 1 and 2.

   if ( i_key == 0 || ! i_key->isString() || i_value == 0 ||
        ( i_section != 0 && ! i_section->isString() && ! i_section->isNil() )
      )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S, S" ) );
   }

   if ( i_section != 0 && i_section->isNil() )
      i_section = 0;

   String *value;
   bool delValue;

   if( i_value->isArray() )
   {
      CoreArray *array = i_value->asArray();
      bool first = true;

      for ( uint32 i = 0; i < array->length(); i ++ )
      {
         Item &itm = array->at( i );

         if( itm.isString() )
         {
            delValue = false;
            value = itm.asString();
         }
         else {
            value = new CoreString;
            delValue = true;
            vm->itemToString( *value, &itm );
         }

         if ( first )
         {
            // setValue will remove every previous reference...
            if( i_section == 0 )
               cfile->setValue( *i_key->asString(), *value );
            else
               cfile->setValue( *i_section->asString(), *i_key->asString(), *value );

            first = false;
         }
         else {
            // ...then we can begin to add
            if( i_section == 0 )
               cfile->addValue( *i_key->asString(), *value );
            else
               cfile->addValue( *i_section->asString(), *i_key->asString(), *value );
         }

         if ( delValue )
            delete value;
      }

      // we have no more business here
      return;
   }
   else if( i_value->isString() )
   {
      delValue = false;
      value = i_value->asString();
   }
   else {
      value = new CoreString;
      delValue = true;
      vm->itemToString( *value, i_value );
   }

   if( i_section == 0 )
      cfile->setValue( *i_key->asString(), *value );
   else
      cfile->setValue( *i_section->asString(), *i_key->asString(), *value );

   if ( delValue )
      delete value;
}

/*#
   @method remove ConfParser
   @brief Remove a key from the configuration file..
   @param key The key to be removed.
   @optparam section If provided, the section where to remove the entry.
   @return True if the key is removed, false if the given key is not found.

   Remove all the instances of a given key from the main section,
   or if @b section parameter is given and not nil, from the specified section.

   The method returns true if the section (when provided) and keys were found,
   and false if nothing has actually been deleted.

*/
FALCON_FUNC  ConfParser_remove( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   Item *i_key = vm->param(0);
   Item *i_section = vm->param(1); // optional

   if ( i_key == 0 || ! i_key->isString() ||
         ( i_section != 0 && ! i_section->isString() && ! i_section->isNil() )
      )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S, S" ) );
   }

   if ( i_section == 0 || i_section->isNil() )
   {
      cfile->removeValue( *i_key->asString() );
   }
   else
   {
      cfile->removeValue( *i_section->asString(), *i_key->asString() );
   }
}

/*#
   @method removeCategory ConfParser
   @brief Remove a whole category from the configuration file..
   @param category The category to be removed.
   @optparam section If provided, the section where to remove the entry.

   This method removes all the entries that would be returned by getCategory
   if provided with the same parameters. The function silently returns doing
   nothing if given category, or given section, cannot be found.
*/

FALCON_FUNC  ConfParser_removeCategory( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   Item *i_category = vm->param(0);
   Item *i_section = vm->param(1); // optional

   if ( i_category == 0 || ! i_category->isString() ||
         ( i_section != 0 && ! i_section->isString() && ! i_section->isNil() )
      )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S, S" ) );
   }

   if ( i_section == 0 || i_section->isNil() )
   {
      cfile->removeCategory( *i_category->asString() );
   }
   else
   {
      cfile->removeCategory( *i_section->asString(), *i_category->asString() );
   }

}

/*#
   @method addSection ConfParser
   @brief Adds an empty section to the configuration file.
   @param section The name of the section to be added.
   @return True if the key is removed, false if the given key is not found.

   Adds an empty section to the configuration file, if it was not already present.
   If a section with the given name is present, nothing is done.
*/
FALCON_FUNC  ConfParser_addSection( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   Item *i_section = vm->param(0);

   if ( i_section == 0 ||  ! i_section->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S" ) );
   }

   vm->retval( (int64) ( cfile->addSection( *i_section->asString() ) == 0 ? 0: 1) );
}


/*#
   @method removeSection ConfParser
   @brief Removes a whole section from the configuration file.
   @param section The name of the section to be removed.
   @return True if the section is removed, false if the given section is not found.

   This method removes the given section from the configuration file. All the keys
   contained in the section, comments below the section declaration and the section
   declaration itself are removed. The function returns true if the given section
   can be found, and false otherwise.
*/

FALCON_FUNC  ConfParser_removeSection( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   Item *i_section = vm->param(0);

   if ( i_section == 0 ||  ! i_section->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S" ) );
   }

   vm->retval( (int64) ( cfile->removeSection( *i_section->asString() ) ? 1: 0) );
}


/*#
   @method clearMain ConfParser
   @brief Clears the main section.

   Removes all the entries from the main section.
   Of course, the section itself is not removed.
*/
FALCON_FUNC  ConfParser_clearMain( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ConfigFile *cfile = (ConfigFile *) self->getUserData();
   cfile->clearMainSection();
}

}
}

/* end of confparser_ext.cpp */
