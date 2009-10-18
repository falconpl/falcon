/*
   The Falcon Programming Language
   FILE: dynlib_mod.h

   Direct dynamic library interface for Falcon
   Interface extension functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 28 Oct 2008 22:23:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: The Falcon Comittee

   See the LICENSE file distributed with this package for licensing details.
*/

/** \file
   Direct dynamic library interface for Falcon
   Internal logic functions - declarations.
*/

#ifndef dynlib_mod_H
#define dynlib_mod_H

#include <falcon/setup.h>
#include <falcon/falcondata.h>
#include <falcon/error.h>
#include <falcon/autocstring.h>
#include <falcon/autowstring.h>

namespace Falcon
{

class Parameter;
class Tokenizer;

class ParamList
{
public:
   ParamList();
   ParamList( const ParamList& other );
   ~ParamList();

   void add(Parameter* p);
   bool isVaradic() const { return m_bVaradic; }

   Parameter* first() const { return m_head; }
   bool empty() const { return m_size == 0; }
   int size() const { return m_size; }

   String toString() const;

private:
   Parameter* m_head;
   Parameter* m_tail;
   int m_size;
   bool m_bVaradic;

};


class Parameter
{
public:
   typedef enum
   {
      e_void,
      e_char,
      e_wchar_t,
      e_unsigned_char,
      e_signed_char,
      e_short,
      e_unsigned_short,
      e_int,
      e_unsigned_int,
      e_long,
      e_unsigned_long,
      e_long_long,
      e_unsigned_long_long,
      e_float,
      e_double,
      e_long_double,
      e_struct,
      e_union,
      e_enum,
      e_varpar
   } e_integral_type;

   Parameter( e_integral_type ct, bool bConst, const String& name, int pointers = 0, int subs = 0, bool isFunc = false );

   Parameter( e_integral_type ct, bool bConst, const String& name, const String& tag, int pointers = 0, int subs = 0, bool isFunc = false );

   Parameter( const Parameter& other );

   ~Parameter();

   /** Type of this parameter. */
   e_integral_type m_type;

   /** Name of the parameters.

      We don't accept unnamed parameters for practical reasons.
   */
   String m_name;

   /** Tag for tagged types. */
   String m_tag;

   bool m_bConst;

   /** Number of pointer indirections */
   int m_pointers;

   /** Array subscript count (-1 for [] ) */
   int m_subscript;

   /** True if this is a function pointer (returning this type)
    * m_pointers will be zero unless this is a pointer to a function pointer.
    * */
   bool m_isFuncPtr;

   /** If this is a function pointer, it may have one or more function parameters. */
   ParamList m_funcParams;

   Parameter* m_next;

   /** returns the number o indirections (pointers/array subscripts) */
   int indirections() const { return m_pointers + m_subscript != 0 ? 1:0; }

   String toString() const;

   static String typeToString( e_integral_type type );
};



class FunctionDef: public FalconData
{

public:

   FunctionDef():
     m_return(0),
     m_fAddress(0)
     {}

  FunctionDef( const String& definition ):
     m_return(0),
     m_fAddress(0)
   {
      parse( definition );
   }

   FunctionDef( const FunctionDef& other );
   virtual ~FunctionDef();


   /** Overrides from falcon data.
      Actually, does nothing.
   */
   virtual void gcMark( uint32 );


   /** Overrides from falcon data.
      Supports complete cloning of the underlying data.
   */
   virtual FalconData *clone() const;

   /* Calls the function address as required by the VM.
      In case the function has a parameter mask, the parameters
      stored in the VM are checked for correct type, and a
      parameter error is raised on mismatch.

      Otherwise, the parameters are turned into integers,
      strings or raw data pointers depending on their type.

      A parameter mismatch may crash the application.

      If a return mask is set for this FunctionPointer, the return
      value from the called function is stored as required in the
      VM return item, otherwise it is discarded.
      \param vm The VM from which parameters are taken and where return values are stored.
      \param firstPar the number of the first parameter in the parameter list that should be
          passed to the function (some parameters may have been used by the Falcon language
          call wrapper).
   */
   //void call( VMachine *vm, int32 firstPar = 0 ) const;

   /** Return this symbol's name */
   const String &name() const { return m_name; }

   /** Parses a string definition.
    * Throws ParseError* on error.
    */
   bool parse( const String& definition );

   /** Returns a representation of this function */
   String toString() const;

   void setFunctionPtr( void* ptr ) { m_fAddress = ptr; }
   void* functionPtr() const { return m_fAddress; }

   const ParamList& params() const { return m_params; }
   ParamList& params() { return m_params; }

private:

   Parameter* parseNextParam( Tokenizer& tok, bool isFuncName = false);
   void parseFuncParams( ParamList& params, Tokenizer& tok );

   String m_definition;
   String m_name;
   Parameter* m_return;
   ParamList m_params;

   /**
      Function pointer
   */
   void *m_fAddress;
};

class ParamValueList;

/** Single parameter value.
 *
 * This class is meant to transform a Falcon item, or some
 * user provided value, into a byte buffer that can be
 * immediately stored in the stack, or in a register, and
 * used in machine-level calls.
 *
 * This class behavior can be slightly architecture dependent;
 * for example, the way to store float numbers and the size of
 * void pointers change across architectures.
 */
class ParamValue: public BaseAlloc
{
public:
   /** Creates a parameter value.
    * If given a parameter type to which refer to,
    * it will be used during transformations.
    *
    * Otherwise, the transformation from falcon Item will use
    * default settings.
    */
   ParamValue( Parameter* type=0 );
   ~ParamValue();

   /** Transform a falcon Item value.
    *
    * It uses the type information to determine if the transformation
    * is possible, and how to perform it at bit-wise level.
    *
    * If the transformation is impossible either because of incompatible
    * types or because of unsupported translations, returns false.
    */
   bool transform( const Item& item );

   /** Prepares to store a void* */
   void transform( void* value )  {
      m_buffer.vptr = value;
      m_size = sizeof( void* );
   }

   /** Prepares to store a long long */
   void transform( int64 value ) {
      m_buffer.vint64 = value;
      m_size = sizeof( int64 );
   }

   /** Prepares to store an int long */
   void transform( int value ) {
      m_buffer.vint = value;
      m_size = sizeof( int );
   }

   /** Prepares to store an int long */
   void transform( unsigned int value ) {
      m_buffer.vint = (int) value;
      m_size = sizeof( int );
   }

   void transform( long value ) {
      m_buffer.vlong = value;
      m_size = sizeof( int );
   }

   void transform( unsigned long value ) {
      m_buffer.vlong = (long) value;
      m_size = sizeof( int );
   }


   /** Prepares to store a char */
   void transform( char value ) {
      m_buffer.vint = (int) value;
      m_size = sizeof( int );
   }

   /** Prepares to store a char */
   void transform( unsigned char value ) {
      m_buffer.vint = (unsigned int) value;
      m_size = sizeof( int );
   }

   /** Prepares to store a float */
   void transform( float value ) {
      m_buffer.vfloat = value;
      m_size = 0x80 | sizeof( float );
   }

   /** Prepares to store a double */
   void transform( double value ) {
      m_buffer.vdouble = value;
      m_size = 0x80 | sizeof( double );
   }

   /** Prepares to store a long double */
   void transform( long double value ) {
      m_buffer.vld = value;
      m_size = 0x80 | sizeof( long double );
   }

   /** Prepare to store an UTF-8 CString representation */
   void makeCString( const String& value );

   /** Prepare to store a wchar_t* representation */
   void makeWString( const String& value );

   const byte* buffer() const { return m_buffer.vbuffer; }
   int32 size() const { return m_size; }

   Parameter* parameter() const { return m_param; }

private:
   friend class ParamValueList;

   // parameter to which we refer to.
   Parameter* m_param;

   /** buffer where to store the transformed value. */
   union tag_m_buffer
   {
      byte vbuffer[16];
      void* vptr;
      int   vint;
      long  vlong;
      int64 vint64;
      float vfloat;
      double vdouble;
      long double vld;
   } m_buffer;

   /* Size in bytes of the transformed value */
   uint32 m_size;

   /** places where to store the transformed strings */
   AutoCString* m_cstring;

   /** places where to store the transformed strings */
   AutoWString* m_wstring;

   /** next parameter value in a chain. */
   ParamValue* m_next;

   bool transformWithParam( const Item& item );
   bool transformFree( const Item& item );
};

/** List of Parameter Values.
 *
 * The list is compiled with parameters as they are loaded in the dynlib call.
 * Once complete, the parameter list creates a pair of parallel arrays
 * that are used by the system-specific machine level call generator.
 *
 * The two arrays store:
 * - Address of parameter N
 * - Size in bytes(4-8-16) and "type" (integer/float) of parameter N
 *
 * So, the first array is an array of void* pointing to the parameter location,
 * and the second array is a sequence of integer values that can be used to
 * determine how much space to push, or what parameter to take.
 *
 * About the parameter type, float parameters are separately stored in
 * stack or in registers under some platforms, so their different nature
 * must be known.
 *
 * Float parameters have the 0x80 bit set.
 *
 */

class ParamValueList: public BaseAlloc
{
public:
   ParamValueList();
   ~ParamValueList();

   /** Adds another parameter value. */
   void add( ParamValue* v );

   /** Creates the list of pointers to the parameters. */
   void compile();

   void** params() const { return m_compiledParams; }
   int* sizes() const { return m_compiledSizes; }
   uint32 count() const { return m_size; }

private:

   ParamValue* m_head;
   ParamValue* m_tail;

   uint32 m_size;
   void** m_compiledParams;
   int* m_compiledSizes;

};

/**
 * Error for all DynLib errors.
 */
class DynLibError: public ::Falcon::Error
{
public:
   DynLibError():
      Error( "DynLibError" )
   {}

   DynLibError( const ErrorParam &params  ):
      Error( "DynLibError", params )
      {}
};

}

#endif

/* end of dynlib_mod.h */
