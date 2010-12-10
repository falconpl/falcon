/**
 *  \file gtk_CellRendererProgress.cpp
 */

#include "gtk_CellRendererProgress.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void CellRendererProgress::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_CellRendererProgress = mod->addClass( "GtkCellRendererProgress", &CellRendererProgress::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkCellRenderer" ) );
    c_CellRendererProgress->getClassDef()->addInheritance( in );

    //c_CellRendererProgress->setWKS( true );
    c_CellRendererProgress->getClassDef()->factory( &CellRendererProgress::factory );

}


CellRendererProgress::CellRendererProgress( const Falcon::CoreClass* gen, const GtkCellRendererProgress* renderer )
    :
    Gtk::CoreGObject( gen, (GObject*) renderer )
{}


Falcon::CoreObject* CellRendererProgress::factory( const Falcon::CoreClass* gen, void* renderer, bool )
{
    return new CellRendererProgress( gen, (GtkCellRendererProgress*) renderer );
}


/*#
    @class GtkCellRendererProgress
    @brief Renders numbers as progress bars

    GtkCellRendererProgress renders a numeric value as a progress par in a cell.
    Additionally, it can display a text on top of the progress bar.
 */
FALCON_FUNC CellRendererProgress::init( VMARG )
{
    NO_ARGS
    MYSELF;
    self->setObject( (GObject*) gtk_cell_renderer_progress_new() );
}


} // Gtk
} // Falcon
