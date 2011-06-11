/**
 *  \file gtk_CellRendererPixbuf.cpp
 */

#include "gtk_CellRendererPixbuf.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void CellRendererPixbuf::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_CellRendererPixbuf = mod->addClass( "GtkCellRendererPixbuf", &CellRendererPixbuf::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkCellRenderer" ) );
    c_CellRendererPixbuf->getClassDef()->addInheritance( in );

    //c_CellRendererPixbuf->setWKS( true );
    c_CellRendererPixbuf->getClassDef()->factory( &CellRendererPixbuf::factory );

}


CellRendererPixbuf::CellRendererPixbuf( const Falcon::CoreClass* gen, const GtkCellRendererPixbuf* renderer )
    :
    Gtk::CoreGObject( gen, (GObject*) renderer )
{}


Falcon::CoreObject* CellRendererPixbuf::factory( const Falcon::CoreClass* gen, void* renderer, bool )
{
    return new CellRendererPixbuf( gen, (GtkCellRendererPixbuf*) renderer );
}


/*#
    @class GtkCellRendererPixbuf
    @brief Renders a pixbuf in a cell

    A GtkCellRendererPixbuf can be used to render an image in a cell. It allows
    to render either a given GdkPixbuf (set via the pixbuf property) or a stock
    icon (set via the stock-id property).

    To support the tree view, GtkCellRendererPixbuf also supports rendering two
    alternative pixbufs, when the is-expander property is TRUE. If the
    is-expanded property is TRUE and the pixbuf-expander-open property is set to
    a pixbuf, it renders that pixbuf, if the is-expanded property is FALSE and
    the pixbuf-expander-closed property is set to a pixbuf, it renders that one.
 */
FALCON_FUNC CellRendererPixbuf::init( VMARG )
{
    NO_ARGS
    MYSELF;
    self->setObject( (GObject*) gtk_cell_renderer_pixbuf_new() );
}


} // Gtk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
