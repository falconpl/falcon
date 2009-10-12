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


namespace Falcon {
#define F_DYNLIB_MAX_PARAMS   32

// we need some old-type define to be sure to have one-byte specificators.
/** End of parameters */
#define F_DYNLIB_PTYPE_END   0
/** P */
#define F_DYNLIB_PTYPE_PTR    1
/** F */
#define F_DYNLIB_PTYPE_FLOAT  2
/** D */
#define F_DYNLIB_PTYPE_DOUBLE 3
/** I */
#define F_DYNLIB_PTYPE_I32    4
/** U */
#define F_DYNLIB_PTYPE_U32    6
/** L */
#define F_DYNLIB_PTYPE_LI     7
/** S */
#define F_DYNLIB_PTYPE_SZ     8
/** W */
#define F_DYNLIB_PTYPE_WZ     9
/** M */
#define F_DYNLIB_PTYPE_MB     10
/** Anything else */
#define F_DYNLIB_PTYPE_OPAQUE 11
/** ... */
#define F_DYNLIB_PTYPE_VAR    12
/** By pointer? */
#define F_DYNLIB_PTYPE_BYPTR  0x80



class ParamType: public BaseAlloc
{
public:
   typedef enum tag_t_type {
      e_void,
      e_char,
      e_uchar,
      e_short,
      e_ushort,
      e_int,
      e_uint,
      e_long,
      e_ulong,
      e_llong,
      e_ullong,
      e_float,
      e_double,
      e_ldouble,
      e_structptr,
      e_funcptr,
      e_ptr
   } t_type;

   typedef enum tag_t_mode {
      IN,
      OUT,
      INOUT
   } t_mode;

   t_type type() const { return m_type; }
   t_mode mode() const { return m_mode; }

   bool isConst() const { return m_bConst; }
   bool isPtr() const { return m_bPtr; }
   bool isVector() const { return m_bVector; }

   const String& name() const { return m_name; }

private:
   t_type m_type;
   t_mode m_mode;

   bool m_bConst;
   bool m_bPtr;
   bool m_bVector;

   Falcon::String m_name;
};


class ParamValue: public BaseAlloc
{
public:
   typedef union tag_u_content {
      char v_char;
      unsigned char v_uchar;
      short v_short;
      unsigned short v_ushort;
      int v_int;
      unsigned int v_uint;
      long v_long;
      unsigned long v_ulong;
      int64 v_llong;
      uint64 v_ullong;
      float v_float;
      double v_double;
      long double v_ldouble;
      void* v_ptr;
   } u_content;

   ParamValue():
      m_ptype( ParamType::e_void )
      {}

   ParamValue( const ParamValue &other ):
      m_ptype( other.m_ptype ),
      m_content( other.m_content )
   {}

   ParamValue( char p_char ):
      m_ptype( ParamType::e_char )
   {
      m_content.v_char = p_char;
   }

   ParamValue( unsigned char p_char ):
      m_ptype( ParamType::e_uchar )
   {
      m_content.v_uchar = p_char;
   }

   ParamValue( short p_short ):
      m_ptype( ParamType::e_short )
   {
      m_content.v_short = p_short;
   }

   ParamValue( unsigned short p_short ):
      m_ptype( ParamType::e_ushort )
   {
      m_content.v_ushort = p_short;
   }

   ParamValue( int p_int ):
      m_ptype( ParamType::e_int )
   {
      m_content.v_int = p_int;
   }

   ParamValue( unsigned int p_int ):
      m_ptype( ParamType::e_uint )
   {
      m_content.v_uint = p_int;
   }

   ParamValue( long p_long ):
     m_ptype( ParamType::e_long )
   {
     m_content.v_long = p_long;
   }

   ParamValue( unsigned long p_long ):
     m_ptype( ParamType::e_ulong )
   {
     m_content.v_ulong = p_long;
   }

   ParamValue( int64 p_llong ):
     m_ptype( ParamType::e_llong )
   {
     m_content.v_llong = p_llong;
   }

   ParamValue( uint64 p_ullong ):
     m_ptype( ParamType::e_ullong )
   {
     m_content.v_ullong = p_ullong;
   }

   ParamValue( float p_float ):
     m_ptype( ParamType::e_float )
   {
     m_content.v_float = p_float;
   }

   ParamValue( double p_double ):
     m_ptype( ParamType::e_double )
   {
     m_content.v_double = p_double;
   }

   ParamValue( long double p_ld ):
     m_ptype( ParamType::e_ldouble )
   {
     m_content.v_ldouble = p_ld;
   }

   ParamValue( void* v_ptr ):
     m_ptype( ParamType::e_ptr )
   {
     m_content.v_ptr = v_ptr;
   }

   void content( char p_char )
   {
      m_ptype = ParamType::e_char;
      m_content.v_char = p_char;
   }

   void content( unsigned char p_char )
   {
      m_ptype = ParamType::e_uchar;
      m_content.v_uchar = p_char;
   }

   void content( short p_short )
   {
      m_ptype = ParamType::e_short;
      m_content.v_short = p_short;
   }

   void content( unsigned short p_short )
   {
      m_ptype = ParamType::e_ushort;
      m_content.v_ushort = p_short;
   }

   void content( int p_int )
   {
      m_ptype = ParamType::e_int;
      m_content.v_int = p_int;
   }

   void content( unsigned int p_int )
   {
      m_ptype = ParamType::e_uint;
      m_content.v_uint = p_int;
   }

   void content( long p_long )
   {
      m_ptype = ParamType::e_long;
      m_content.v_long = p_long;
   }

   void content( unsigned long p_long )
   {
      m_ptype = ParamType::e_ulong;
      m_content.v_ulong = p_long;
   }

   void content( int64 p_llong )
   {
      m_ptype = ParamType::e_llong;
      m_content.v_llong = p_llong;
   }

   void content( uint64 p_ullong )
   {
      m_ptype = ParamType::e_ullong;
      m_content.v_ullong = p_ullong;
   }

   void content( float p_float )
   {
      m_ptype = ParamType::e_float;
      m_content.v_float = p_float;
   }

   void content( double p_double )
   {
      m_ptype = ParamType::e_double;
      m_content.v_double = p_double;
   }

   void content( long double p_ld )
   {
      m_ptype = ParamType::e_ldouble;
      m_content.v_ldouble = p_ld;
   }

   void content( void* v_ptr )
   {
      m_ptype = ParamType::e_ptr;
      m_content.v_ptr = v_ptr;
   }

   u_content content() const { return m_content; }

private:
   ParamType::t_type m_ptype;
   u_content m_content;
};


class FunctionDef: public FalconData
{
   String m_definition;
   String m_name;
   ParamType m_return;
   ParamType *m_params;
   uint32 m_paramCount;

public:

   FunctionDef();

   FunctionDef( const String& definition ):
      m_params(0),
      m_paramCount(0)
   {
      parse( definition );
   }

   FunctionDef( const FunctionDef& other );
   virtual ~FunctionDef();

   bool isDeclared() const { return m_paramCount != 0xFFFFFFFF; }

   /** Parses a string definition.
    * Throws ParseError* on error.
    */
   void parse( const String& definition );

   int paramCount() const { return m_paramCount; }
   ParamType* params() const { return m_params; }
   const ParamType& returnType() const { return m_return; }
   const String& name() const { return m_name; }
   const String& definition() const { return m_definition; }

};

class FunctionAddress: public FalconData
{
   String m_name;
   byte *m_parsedParams;
   byte m_parsedReturn;

   /** Array of strings parallel to m_parsedParams where safety input types are stored. */
   String *m_safetyParams;

   uint32 m_parsedParamsCount;
   uint32 m_safetyParamsCount;

public:
   /**
      Function pointer
   */
   void *m_fAddress;

   /**
      The (original) parameter mask.
   */
   String m_paramMask;

   /**
      The return mask.
   */
   String m_returnMask;

   /** Should we guess our params? */
   bool m_bGuessParams;

   FunctionAddress( const String &name, void *address = 0 ):
      m_name( name ),
      m_parsedParams(0),
      m_parsedReturn( F_DYNLIB_PTYPE_END ),
      m_safetyParams(0),
      m_parsedParamsCount(0),
      m_safetyParamsCount(0),
      m_fAddress(address),
      m_bGuessParams( true )
   {}

   /**
      Copies an existing function pointer.
      /TODO
   */
   FunctionAddress( const FunctionAddress &other ):
      m_parsedParams(0),
      m_parsedReturn( other.m_parsedReturn ),
      m_safetyParams(0),
      m_parsedParamsCount(0),
      m_safetyParamsCount(0),
      m_fAddress( other.m_fAddress ),
      m_paramMask( other.m_paramMask ),
      m_returnMask( other.m_returnMask ),
      m_bGuessParams( other.m_bGuessParams )
   {}

   // nothing needed to be done.
   virtual ~FunctionAddress();

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

   /** Parse and eventually setup parameter masks.

      The parameter mask is parsed scanning a string containing tokens
      separated by whitespaces, ',' or ';' (they are the same).

      Each parameter specificator is either a single character or a "pseudoclass" name,
      which must be an arbitrary name long two characters more.

      The single character parameter specificator may be one of the following:

      - P - application specific opaque pointer (stored in a Falcon integer item).
      - F - 32 bit IEEE float format.
      - D - 64 bit IEEE double format.
      - I - 32 bit signed integer. This applies also to char and short int types, which
            are always padded to 32 bit integers when passed as parameters or returned.
      - U - 32 bit unsigned integer. This applies also to bytes and short unsigned int types, which
            are always padded to 32 bit integers when passed as parameters or returned.
      - L - 64 bit integers; as this is the maximum size allowed, sign is not relevant (the sign bit
            is always placed correctly both in parameter passing and return values).
      - S - UTF-8 encoded strings.
      - W - Wide character strings (compatible with UTF-16 in local byte ordering).
      - M - Memory buffers (MemBuf); this may also contain natively encoded strings.

      The special "..." token indicates that the function accepts a set of unknown
      parameters after that point, which will be treated as in unsafe mode.

      A pseudoclass name serves as a type safety constarint for the loaded library. Return
      values marked with pseudoclasses will generate a DynOpaque instance that will remember
      the class name and wrap the transported item. A pseudoclass parameter will check for the
      parameter passed by the Falcon script at the given position is of class DynOpaque and carrying
      the required pseudoclass type.

      Prepending a '$' sign in front of the parameter specificator will inform the parameter parsing
      system to pass the item by pointer to the underlying library.
      The Falcon script must pass the coresponding  parameter by reference, and after the underlying
      function returns, a possibly set-up or modified value is taken and stored into the parameter.
      Return specifiers cannot be prepended with '$'.

      \note Ideographic language users can define pseudoclasses with a single ideographic char,
      as a pseudoclass name is parsed if the count of characters in a substring is more than one,
      or if the only character is > 256U.
   */
   bool parseParams( const String &params );

   /**
      Parses the return value type specifier.

      On success, also configures this object's internal data.
   */
   bool parseReturn( const String &rval );

   /** Parses a single parameter, finding its type.
      \param mask A parameter list format.
      \param type gets the output type.
      \param begin First character in the mask to be parsed.
      \param end last character in the mask to be parsed.
      \return false if the parameter list is malformed, true (and type filled) on succesful parsing.
   */
   bool parseSingleParam( const String &mask, byte &type, uint32 begin=0, uint32 end=String::npos );

   /**
      Return the decoded nth param.
   */
   byte parsedParam( uint32 id ) const { return m_parsedParams[id]; }
   uint32 parsedParamCount() const { return m_parsedParamsCount; }

   /** Return the nth pseudoclass name being present in the parsed parameters.
      If pid if out of range or the parameters are not parsed, the function will crash.
   */
   const String &pclassParam( uint32 pid ) const { return m_safetyParams[pid]; }
   uint32 pclassCount() const { return m_safetyParamsCount; }

   byte parsedReturn() const { return m_parsedReturn; }
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
