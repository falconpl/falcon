/*
 *  Falcon DataMatrix - Module definition
 */

#include "dmtx_ext.h"
#include "dmtx_mod.h"
#include "dmtx_srv.h"
//#include "dmtx_st.h"
#include "version.h"

#include <falcon/module.h>

#include <dmtx.h>

/*#
   @module dmtx DataMatrix module
   @brief DataMatrix utilities.
 */

/*
    TODO LIST...
 */

using namespace Falcon;

Falcon::DataMatrixService theDataMatrixService;


FALCON_MODULE_DECL
{
    Falcon::Module *self = new Falcon::Module();
    self->name( "dmtx" );
    self->engineVersion( FALCON_VERSION_NUM );
    self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

    //#include "dmtx_st.h"

    // *** Constants ***
    self->addConstant( "DmtxVersion",           DmtxVersion );
    self->addConstant( "DmtxUndefined",         (int64) DmtxUndefined );
    // Schemes...
    self->addConstant( "DmtxSchemeAutoFast",    (int64) DmtxSchemeAutoFast );
    self->addConstant( "DmtxSchemeAutoBest",    (int64) DmtxSchemeAutoBest );
    self->addConstant( "DmtxSchemeAscii",       (int64) DmtxSchemeAscii );
    self->addConstant( "DmtxSchemeC40",         (int64) DmtxSchemeC40 );
    self->addConstant( "DmtxSchemeText",        (int64) DmtxSchemeText );
    self->addConstant( "DmtxSchemeX12",         (int64) DmtxSchemeX12 );
    self->addConstant( "DmtxSchemeEdifact",     (int64) DmtxSchemeEdifact );
    self->addConstant( "DmtxSchemeBase256",     (int64) DmtxSchemeBase256 );
    // Symbol sizes...
    self->addConstant( "DmtxSymbolRectAuto",    (int64) DmtxSymbolRectAuto );
    self->addConstant( "DmtxSymbolSquareAuto",  (int64) DmtxSymbolSquareAuto );
    self->addConstant( "DmtxSymbolShapeAuto",   (int64) DmtxSymbolShapeAuto );
    self->addConstant( "DmtxSymbol10x10",       (int64) DmtxSymbol10x10 );
    self->addConstant( "DmtxSymbol12x12",       (int64) DmtxSymbol12x12 );
    self->addConstant( "DmtxSymbol14x14",       (int64) DmtxSymbol14x14 );
    self->addConstant( "DmtxSymbol16x16",       (int64) DmtxSymbol16x16 );
    self->addConstant( "DmtxSymbol18x18",       (int64) DmtxSymbol18x18 );
    self->addConstant( "DmtxSymbol20x20",       (int64) DmtxSymbol20x20 );
    self->addConstant( "DmtxSymbol22x22",       (int64) DmtxSymbol22x22 );
    self->addConstant( "DmtxSymbol24x24",       (int64) DmtxSymbol24x24 );
    self->addConstant( "DmtxSymbol26x26",       (int64) DmtxSymbol26x26 );
    self->addConstant( "DmtxSymbol32x32",       (int64) DmtxSymbol32x32 );
    self->addConstant( "DmtxSymbol36x36",       (int64) DmtxSymbol36x36 );
    self->addConstant( "DmtxSymbol40x40",       (int64) DmtxSymbol40x40 );
    self->addConstant( "DmtxSymbol44x44",       (int64) DmtxSymbol44x44 );
    self->addConstant( "DmtxSymbol48x48",       (int64) DmtxSymbol48x48 );
    self->addConstant( "DmtxSymbol52x52",       (int64) DmtxSymbol52x52 );
    self->addConstant( "DmtxSymbol64x64",       (int64) DmtxSymbol64x64 );
    self->addConstant( "DmtxSymbol72x72",       (int64) DmtxSymbol72x72 );
    self->addConstant( "DmtxSymbol80x80",       (int64) DmtxSymbol80x80 );
    self->addConstant( "DmtxSymbol88x88",       (int64) DmtxSymbol88x88 );
    self->addConstant( "DmtxSymbol96x96",       (int64) DmtxSymbol96x96 );
    self->addConstant( "DmtxSymbol104x104",     (int64) DmtxSymbol104x104 );
    self->addConstant( "DmtxSymbol120x120",     (int64) DmtxSymbol120x120 );
    self->addConstant( "DmtxSymbol132x132",     (int64) DmtxSymbol132x132 );
    self->addConstant( "DmtxSymbol144x144",     (int64) DmtxSymbol144x144 );
    self->addConstant( "DmtxSymbol8x18",        (int64) DmtxSymbol8x18 );
    self->addConstant( "DmtxSymbol8x32",        (int64) DmtxSymbol8x32 );
    self->addConstant( "DmtxSymbol12x26",       (int64) DmtxSymbol12x26 );
    self->addConstant( "DmtxSymbol12x36",       (int64) DmtxSymbol12x36 );
    self->addConstant( "DmtxSymbol16x36",       (int64) DmtxSymbol16x36 );
    self->addConstant( "DmtxSymbol16x48",       (int64) DmtxSymbol16x48 );
    // Directions...
    self->addConstant( "DmtxDirNone",           (int64) DmtxDirNone );
    self->addConstant( "DmtxDirUp",             (int64) DmtxDirUp );
    self->addConstant( "DmtxDirLeft",           (int64) DmtxDirLeft );
    self->addConstant( "DmtxDirDown",           (int64) DmtxDirDown );
    self->addConstant( "DmtxDirRight",          (int64) DmtxDirRight );
    self->addConstant( "DmtxDirHorizontal",     (int64) DmtxDirHorizontal );
    self->addConstant( "DmtxDirVertical",       (int64) DmtxDirVertical );
    self->addConstant( "DmtxDirRightUp",        (int64) DmtxDirRightUp );
    self->addConstant( "DmtxDirLeftDown",       (int64) DmtxDirLeftDown );

    // DataMatrix class
    Falcon::Symbol* dmtx_cls = self->addClass( "DataMatrix" );
    dmtx_cls->setWKS( true );
    dmtx_cls->getClassDef()->factory( Falcon::Dmtx::DataMatrix::factory );

    self->addClassProperty( dmtx_cls, "module_size" );
    self->addClassProperty( dmtx_cls, "margin_size" );
    self->addClassProperty( dmtx_cls, "gap_size" );
    self->addClassProperty( dmtx_cls, "scheme" );
    self->addClassProperty( dmtx_cls, "shape" );

    self->addClassProperty( dmtx_cls, "timeout" );
    self->addClassProperty( dmtx_cls, "shrink" );
    self->addClassProperty( dmtx_cls, "deviation" );
    self->addClassProperty( dmtx_cls, "threshold" );
    self->addClassProperty( dmtx_cls, "min_edge" );
    self->addClassProperty( dmtx_cls, "max_edge" );
    self->addClassProperty( dmtx_cls, "corrections" );
    self->addClassProperty( dmtx_cls, "max_count" );

    self->addClassMethod( dmtx_cls, "encode", Falcon::Ext::DataMatrix_encode );
    self->addClassMethod( dmtx_cls, "decode", Falcon::Ext::DataMatrix_decode );
    self->addClassMethod( dmtx_cls, "resetOptions", Falcon::Ext::DataMatrix_resetOptions );

    // DataMatrixError class
    Falcon::Symbol *error_class = self->addExternalRef( "Error" ); // it's external
    Falcon::Symbol *err_cls = self->addClass( "DataMatrixError", &Falcon::Ext::DataMatrixError_init );
    err_cls->setWKS( true );
    err_cls->getClassDef()->addInheritance( new Falcon::InheritDef( error_class ) );

    // service publication
    self->publishService( &theDataMatrixService );

    return self;
}
