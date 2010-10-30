/*
   FALCON - The Falcon Programming Language.
   FILE: bufext_ext.cpp

   Buffering extensions
   Main module file, providing the module object to
   the Falcon engine.
   -------------------------------------------------------------------
   Author: Maximilian Malek
   Begin: Sun, 20 Jun 2010 18:59:55 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: The above AUTHOR

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

#include <falcon/module.h>
#include <falcon/symbol.h>
#include "bufext_ext.h"
#include "bufext_ext.inl"
#include "bufext_st.h"

#include "../include/version.h"

#include "bytebuf.h"
#include "bitbuf.h"


/*#
   @main bufext

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

template <typename BUFTYPE> Falcon::Symbol *SimpleRegisterBuf(Falcon::Module *self, const char *name)
{
    Falcon::Symbol *cls = self->addClass(name, Falcon::Ext::Buf_init<BUFTYPE>);
    self->addClassMethod(cls, OVERRIDE_OP_GETINDEX, Falcon::Ext::Buf_getIndex<BUFTYPE>);
    self->addClassMethod(cls, OVERRIDE_OP_SETINDEX, Falcon::Ext::Buf_setIndex<BUFTYPE>);
    self->addClassMethod(cls, "setEndian", Falcon::Ext::Buf_setEndian<BUFTYPE>);
    self->addClassMethod(cls, "getEndian", Falcon::Ext::Buf_getEndian<BUFTYPE>);
    self->addClassMethod(cls, "size", Falcon::Ext::Buf_size<BUFTYPE>);
    self->addClassMethod(cls, "resize", Falcon::Ext::Buf_resize<BUFTYPE>);
    self->addClassMethod(cls, "reserve", Falcon::Ext::Buf_reserve<BUFTYPE>);
    self->addClassMethod(cls, "capacity", Falcon::Ext::Buf_capacity<BUFTYPE>);
    self->addClassMethod(cls, "readable", Falcon::Ext::Buf_readable<BUFTYPE>);
    self->addClassMethod(cls, "growable", Falcon::Ext::Buf_growable<BUFTYPE>);
    self->addClassMethod(cls, "wpos", Falcon::Ext::Buf_wpos<BUFTYPE>);
    self->addClassMethod(cls, "rpos", Falcon::Ext::Buf_rpos<BUFTYPE>);
    self->addClassMethod(cls, "reset", Falcon::Ext::Buf_reset<BUFTYPE>);
    self->addClassMethod(cls, "write", Falcon::Ext::Buf_write<BUFTYPE, true>);
    self->addClassMethod(cls, "writeNoNT", Falcon::Ext::Buf_write<BUFTYPE, false>);
    self->addClassMethod(cls, "writePtr", Falcon::Ext::Buf_writePtr<BUFTYPE>).asSymbol()
        ->addParam("ptr")->addParam("bytes");
    self->addClassMethod(cls, "readString", Falcon::Ext::Buf_readString<BUFTYPE>).asSymbol()
        ->addParam("charSize");
    self->addClassMethod(cls, "readToBuf", Falcon::Ext::Buf_readToBuf<BUFTYPE>).asSymbol()
        ->addParam("bytes");
    self->addClassMethod(cls, "readPtr", Falcon::Ext::Buf_readPtr<BUFTYPE>).asSymbol()
        ->addParam("ptr")->addParam("bytes");
    self->addClassMethod(cls, "toMemBuf", Falcon::Ext::Buf_toMemBuf<BUFTYPE>);
    self->addClassMethod(cls, "ptr", Falcon::Ext::Buf_ptr<BUFTYPE>);
    self->addClassMethod(cls, "toString", Falcon::Ext::Buf_toString<BUFTYPE>);

    self->addClassMethod(cls, "wb", Falcon::Ext::Buf_wb<BUFTYPE>);
    self->addClassMethod(cls, "w8", Falcon::Ext::Buf_w8<BUFTYPE>);
    self->addClassMethod(cls, "w16", Falcon::Ext::Buf_w16<BUFTYPE>);
    self->addClassMethod(cls, "w32", Falcon::Ext::Buf_w32<BUFTYPE>);
    self->addClassMethod(cls, "w64", Falcon::Ext::Buf_w64<BUFTYPE>);
    self->addClassMethod(cls, "wf", Falcon::Ext::Buf_wf<BUFTYPE>);
    self->addClassMethod(cls, "wd", Falcon::Ext::Buf_wd<BUFTYPE>);
    self->addClassMethod(cls, "rb", Falcon::Ext::Buf_rb<BUFTYPE>);
    self->addClassMethod(cls, "r8", Falcon::Ext::Buf_r8<BUFTYPE>);
    self->addClassMethod(cls, "r16", Falcon::Ext::Buf_r16<BUFTYPE>);
    self->addClassMethod(cls, "r32", Falcon::Ext::Buf_r32<BUFTYPE>);
    self->addClassMethod(cls, "r64", Falcon::Ext::Buf_r64<BUFTYPE>);
    self->addClassMethod(cls, "rf", Falcon::Ext::Buf_rf<BUFTYPE>);
    self->addClassMethod(cls, "rd", Falcon::Ext::Buf_rd<BUFTYPE>);

    return cls;
}

Falcon::Module *bufext_module_init(void)
{
   #define FALCON_DECLARE_MODULE self

   // initialize the module
   Falcon::Module *self = new Falcon::Module();
   self->name( "bufext" );
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   //============================================================
   // Here declare the international string table implementation
   //
   #include "bufext_st.h"

   //============================================================
   // API declarations
   //
   SimpleRegisterBuf<Falcon::ByteBuf>              (self, "ByteBuf");
   SimpleRegisterBuf<Falcon::ByteBufNativeEndian>  (self, "ByteBufNativeEndian");
   SimpleRegisterBuf<Falcon::ByteBufLittleEndian>  (self, "ByteBufLittleEndian");
   SimpleRegisterBuf<Falcon::ByteBufBigEndian>     (self, "ByteBufBigEndian");
   SimpleRegisterBuf<Falcon::ByteBufReverseEndian> (self, "ByteBufReverseEndian");

   Falcon::Symbol *bitcls = SimpleRegisterBuf<Falcon::BitBuf>(self, "BitBuf");
   self->addClassMethod(bitcls, "bitCount", Falcon::Ext::BitBuf_bitCount);
   self->addClassMethod(bitcls, "writeBits", Falcon::Ext::BitBuf_writeBits);
   self->addClassMethod(bitcls, "readBits", Falcon::Ext::BitBuf_readBits);
   self->addClassMethod(bitcls, "sizeBits", Falcon::Ext::BitBuf_sizeBits);
   self->addClassMethod(bitcls, "rposBits", Falcon::Ext::BitBuf_rposBits);
   self->addClassMethod(bitcls, "wposBits", Falcon::Ext::BitBuf_wposBits);
   self->addClassMethod(bitcls, "readableBits", Falcon::Ext::BitBuf_readableBits);

   // static BitBuf methods
   self->addClassMethod(bitcls, "bitsForInt", Falcon::Ext::BitBuf_bits_req);

   Falcon::Symbol *error_class = self->addExternalRef( "Error" ); // it's external
   Falcon::Symbol *buferr_cls = self->addClass( "BufferError", Falcon::Ext::BufferError_init );
   buferr_cls->setWKS( true );
   buferr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );

   self->addConstant("BUF_NATIVE_ENDIAN",  (Falcon::int64)Falcon::ENDIANMODE_NATIVE);
   self->addConstant("BUF_LITTLE_ENDIAN",  (Falcon::int64)Falcon::ENDIANMODE_LITTLE);
   self->addConstant("BUF_BIG_ENDIAN",     (Falcon::int64)Falcon::ENDIANMODE_BIG);
   self->addConstant("BUF_REVERSE_ENDIAN", (Falcon::int64)Falcon::ENDIANMODE_REVERSE);

   return self;
}

// TODO: remove the debug #ifndef block before pushing !!
#ifndef _DEBUG
FALCON_MODULE_DECL
{
    return bufext_module_init();
}
#endif

/* end of bufext.cpp */
