/*
 * pdf.cpp
 *
 *  Created on: 04.04.2010
 *      Author: maik
 */

#include <falcon/engine.h>
#include <hpdf.h>
#include <scriptExtensions/error.h>
#include <moduleImpl/error.h>

namespace Falcon { namespace Ext { namespace hpdf {

FALCON_FUNC Error::init( VMachine* vm )
{
  CoreObject *einst = vm->self().asObject();
  if( einst->getUserData() == 0 )
     einst->setUserData( new Mod::hpdf::Error );

  core::Error_init( vm );
}

void Error::registerExtensions(Falcon::Module* self)
{
  Falcon::Symbol *errorCls = self->addExternalRef( "Error" ); // it's external
  Falcon::Symbol *hpdfErrCls = self->addClass( "HPDFError", &init );
  hpdfErrCls->getClassDef()->addInheritance(  new Falcon::InheritDef( errorCls ) );
}
}}} // Falcon::Ext::hpdf
