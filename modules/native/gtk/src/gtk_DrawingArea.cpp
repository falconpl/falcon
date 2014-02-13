#include "gtk_DrawingArea.hpp"

namespace Falcon {
namespace Gtk {


void DrawingArea::modInit( Module* mod )
{
    const Symbol* c_DrawingArea = mod->addClass( "GtkDrawingArea", &DrawingArea::init );
    InheritDef* in = new InheritDef( mod->findGlobalSymbol( "GtkWidget") );

    c_DrawingArea->getClassDef()->addInheritance( in );
    c_DrawingArea->setWKS( true );
    c_DrawingArea->getClassDef()->factory( &DrawingArea::factory );

    mod->addClassMethod( c_DrawingArea, "size", &DrawingArea::size );
}


DrawingArea::DrawingArea( const CoreClass* gen, const GtkDrawingArea* area ):
    CoreGObject( gen, (const GObject *) area )
{}


CoreObject* DrawingArea::factory( const CoreClass* gen, void* area, bool )
{
    return new DrawingArea( gen, (GtkDrawingArea*) area );
}


FALCON_FUNC DrawingArea::init( VMARG )
{
    NO_ARGS;
    MYSELF;

    self->setObject( (GObject*) gtk_drawing_area_new() );
}


FALCON_FUNC DrawingArea::size( VMARG )
{
    MYSELF;
    GET_OBJ( self );

    Item* i_width = vm->param( 0 );
    Item* i_height = vm->param( 1 );

#ifndef NO_PARAMETER_CHECK
    if( i_width == 0 || ! i_width->isInteger()
     || i_height == 0 || ! i_height->isInteger() )
    {
        throw_inv_params( "I,I" );
    }
#endif

    gtk_drawing_area_size( (GtkDrawingArea *) _obj, i_width->asInteger(), i_height->asInteger() );
}

}
}


