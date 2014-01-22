/*
   FALCON - The Falcon Programming Language.
   FILE: dbi_bind.h

   Database Interface
   Helper for general Falcon-to-C variable binding
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 May 2010 23:47:36 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_DBI_BIND_H_
#define FALCON_DBI_BIND_H_

#include <falcon/setup.h>
#include <falcon/types.h>

namespace Falcon {

class Item;
class TimeStamp;
class String;
class ItemArray;

/** Time Convert functor.
    This functor is reimplemented by the drivers to allow
    transforming a Falcon TimeStamp class into a local
    timestamp representation.
*/
class DBITimeConverter
{
public:
   inline void operator() ( TimeStamp* ts, void* buffer, int& bufsize ) const
   {
      convertTime( ts, buffer, bufsize );
   }

   /** Sublcasses must re-define this to construct a timestamp that can be used in bindings.

    */
   virtual void convertTime( TimeStamp* ts, void* buffer, int& bufsize ) const = 0;
};

/** Default time converter.
    This time converter tranforms falcon timestamp in ISO timestamps,
    as 1-byte strings encoded like AAAA-MM-GG HH:MM:SS (ignoring milliseconds).
*/
class DBITimeConverter_ISO: public DBITimeConverter
{
public:
   virtual void convertTime( TimeStamp* ts, void* buffer, int& bufsize ) const;
};
extern DBITimeConverter_ISO DBITimeConverter_ISO_impl;


/** String Convert functor.
    This functor is reimplemented by the drivers to allow
    transforming a Falcon string class into a local
    string representation. 
*/
class DBIStringConverter
{
public:
   inline char* operator() ( const String& str, char* target, int &bufsize ) const
   {
      return convertString( str, target, bufsize );
   }

   /** Sublcasses must re-define this to construct a string that can be used in bindings.

    */
   virtual char* convertString( const String& str, char* target, int &bufsize ) const = 0;
};

/** Default string Convert functor for InBind class.

   Returns a string converted into a UTF8 string.
*/
class DBIStringConverter_UTF8: public DBIStringConverter
{
public:
   virtual char* convertString( const String& str, char* target, int &bufsize ) const;
};
extern DBIStringConverter_UTF8 DBIStringConverter_UTF8_impl;


/** Utility string converter functor for InBind class.

   Returns a string converted into a WCHART string.
*/
class DBIStringConverter_WCHAR: public DBIStringConverter
{
public:
   virtual char* convertString( const String& str, char* target, int &bufsize ) const;
};
extern DBIStringConverter_WCHAR DBIStringConverter_WCHAR_impl;

/** Helper class to bind item into local database value.

    This class is used to turn a Falcon item into a C item representation
    which is used by the vast majority of SQL engines to bind
    input variables.

    Creating and populating the final SQL-dependant bind structure
    is up to the engine, but this class simplifies the rutinary
    operations of turning falcon strings, integers, objects and other
    elements into acceptable representations for the final engine.

    To convert the item from a Falcon timestamp object into a DB-engine
    timestamp item, this class uses the DBIBind::convertTime virtual function
    that must be provided by the engine re-implementations.

 */
class DBIBindItem
{
public:
   static const int bufsize = 128;

   DBIBindItem();
   virtual ~DBIBindItem();

   typedef enum tag_datatype
   {
      t_nil,
      t_bool,
      t_int,
      t_double,
      t_string,
      t_time,
      t_buffer
   } datatype;

   void set(const Item& value,
         const DBITimeConverter& tc=DBITimeConverter_ISO_impl,
         const DBIStringConverter& sc=DBIStringConverter_UTF8_impl );

   void clear();

   /** Returns the type of this item. */
   datatype type() const { return m_type; }

   /** Return a void pointer to the stored data. */
   void* data() { return &m_cdata.v_int64; }
   void* databuffer() {
      if(  m_type == t_string || m_type == t_buffer || m_type == t_time )
         return m_cdata.v_buffer;
      return &m_cdata.v_int64;
   }

   bool isNil();
   bool asBool() const { return m_cdata.v_bool; }
   int64 asInteger() const { return m_cdata.v_int64; }
   double asDouble() const { return m_cdata.v_double; }
   const char* asString() const { return m_cdata.v_string; }
   void* asBuffer() const { return m_cdata.v_buffer; }
   int asStringLen() const { return m_buflen; }

   bool* asBoolPtr() { return &m_cdata.v_bool; }
   int64* asIntegerPtr() { return &m_cdata.v_int64; }
   double* asDoublePtr() { return &m_cdata.v_double; }

   /** Gets the user buffer.
       Returns 128 bytes of preallocated memory in this object,
       usually separated by the rest.

       Can be used to store transformation of the main data type,
       for example, local SQL engine renderings of the generic
       timestamp format.
    */
   char* userbuffer() { return m_buffer; }

   /** Returns the inner buffer lenght.

       Valid only in case of strings and timestamps.
    * @return size of the data in the buffer.
    */
   int length() const { return m_buflen; }

private:
   datatype m_type;

   // Local buffer we use for long int and buffers.
   typedef union tag_cdata
   {
      bool v_bool;
      double v_double;
      int64 v_int64;
      char* v_string;
      void* v_buffer;
   } cdata;

   cdata m_cdata;

   // local buffer that can be used for several reasons.
   char m_buffer[bufsize];
   int m_buflen;
};

/** Base abstract class for DBI input bindings.

    Engines must reimplement this class to provide the needed
    callbacks to create their own bind variables.

    The base class creates DBIBindItems generating appropriate
    memory locations where to store the bound input (falcon-to-engine)
    variables.

    The subclasses will receive callbacks when a new binding begins
    and when a binding item had a relevant change (i.e. it's memory
    footprint has changed), so that the change can be reflected in the
    engine specific binding variables.

    The binding fails if a previous binding was already performed, and
    a new binding is tried with different types or with an array of different
    size.

    In that case, an appropriate DBI error is raised.
 */
class DBIInBind
{
public:
   /** Creates a input binding.
    *
    *  Some engines (e.g. sqlite) bind the input buffer BY VALUE; this
    *  requires a complete rebind at each step.
    *
    * @param bAlwaysChange rebind at each step.
    */
   DBIInBind( bool bAlwaysChange = false );
   virtual ~DBIInBind();

   virtual void bind( const ItemArray& arr,
         const DBITimeConverter& tc=DBITimeConverter_ISO_impl,
         const DBIStringConverter& sc=DBIStringConverter_UTF8_impl );

   /*# Bind to 0 parameters */
   void unbind();

   /** Called back when the binding is initialized.
    *
    * @param size The number of the variables that should be allocated.
    */
   virtual void onFirstBinding( int size ) = 0;

   /** Called when an item had a relevant change, requiring reset of the underlying variables.
    *
    * Use the number passed as parameter to get the relevant item in the m_ibinds array.
    * On first call of bind(), this method will be called for every item.
    *
    * @param num Number of the item.
    */
   virtual void onItemChanged( int num ) = 0;

   /** Return true if we're processing the items for the fist time. */
   bool isFirstLoop() const { return m_size == 0; }

protected:
   DBIBindItem* m_ibind;
   bool m_bAlwaysChange;
   int m_size;
};

}

#endif

/* end of dbi_bind.h */
