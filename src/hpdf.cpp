
/**
 * \file
 * This module exports pdf and module loader facility to falcon
 * scripts.
 */

#include <falcon/types.h>
#include <falcon/module.h>

#include <hpdf.h>
#include <hpdf_error.h>
#include "scriptExtensions/consts.h"
#include "scriptExtensions/enums.h"
#include "scriptExtensions/doc.h"
#include "scriptExtensions/page.h"
#include "scriptExtensions/error.h"
#include "scriptExtensions/font.h"
#include "scriptExtensions/image.h"
#include "scriptExtensions/destination.h"
#include "scriptExtensions/outline.h"
#include "scriptExtensions/encoder.h"
#include "scriptExtensions/textannotation.h"
//#include "scriptExtensions/encoder.h"
//#include "scriptExtensions/extgstate.h"
//#include "scriptExtensions/image.h"
//#include "scriptExtensions/linkannot.h"
#include "version.h"
#include "moduleImpl/st.h"


void registerConsts(Falcon::Module*);

FALCON_MODULE_DECL
{
  Falcon::Module *self = new Falcon::Module();
  self->name( "hpdf" );
  self->engineVersion( FALCON_VERSION_NUM );
  self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );
  #define FALCON_DECLARE_MODULE self
  #include "moduleImpl/st.h"
  #undef FALCON_DECLARE_MODULE

  Falcon::Ext::hpdf::registerEnums(self);
  Falcon::Ext::hpdf::registerConsts(self);

  Falcon::Ext::hpdf::Error::registerExtensions(self);
  Falcon::Ext::hpdf::Doc::registerExtensions(self);
  Falcon::Ext::hpdf::Page::registerExtensions(self);
  Falcon::Ext::hpdf::Font::registerExtensions(self);
  Falcon::Ext::hpdf::Destination::registerExtensions(self);
  Falcon::Ext::hpdf::Image::registerExtensions(self);
  Falcon::Ext::hpdf::Outline::registerExtensions(self);
  Falcon::Ext::hpdf::Encoder::registerExtensions(self);
  Falcon::Ext::hpdf::TextAnnotation::registerExtensions(self);
  //Falcon::Ext::hpdf::XObject::registerExtensions(self);

  return self;
}



