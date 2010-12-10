/**
 *  \file gtk_CellRendererText.cpp
 */

#include "gtk_CellRendererText.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void CellRendererText::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_CellRendererText = mod->addClass( "GtkCellRendererText", &CellRendererText::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkCellRenderer" ) );
    c_CellRendererText->getClassDef()->addInheritance( in );

    //c_CellRendererText->setWKS( true );
    c_CellRendererText->getClassDef()->factory( &CellRendererText::factory );

    mod->addClassMethod( c_CellRendererText,
                         "signal_edited",
                         &CellRendererText::signal_edited );
    mod->addClassMethod( c_CellRendererText,
                         "set_fixed_height_from_font",
                         &CellRendererText::set_fixed_height_from_font );
}


CellRendererText::CellRendererText( const Falcon::CoreClass* gen, const GtkCellRendererText* renderer )
    :
    Gtk::CoreGObject( gen, (GObject*) renderer )
{}


Falcon::CoreObject* CellRendererText::factory( const Falcon::CoreClass* gen, void* renderer, bool )
{
    return new CellRendererText( gen, (GtkCellRendererText*) renderer );
}


/*#
    @class GtkCellRendererText
    @brief Renders text in a cell

    A GtkCellRendererText renders a given text in its cell, using the font,
    color and style information provided by its properties. The text will be
    ellipsized if it is too long and the ellipsize property allows it.

    If the mode is GTK_CELL_RENDERER_MODE_EDITABLE, the GtkCellRendererText
    allows to edit its text using an entry.

    Adjust how text is drawn using object properties. Object properties can be
    set globally (with g_object_set()). Also, with GtkTreeViewColumn, you can
    bind a property to a value in a GtkTreeModel. For example, you can bind the
    "text" property on the cell renderer to a string value in the model, thus
    rendering a different string in each row of the GtkTreeView
 */
FALCON_FUNC CellRendererText::init( VMARG )
{
    NO_ARGS
    MYSELF;
    self->setObject( (GObject*) gtk_cell_renderer_text_new() );
}


/*#
    @method signal_edited GtkCellRendererText
    @brief This signal is emitted after renderer has been edited.

    It is the responsibility of the application to update the model and store
    new_text at the position indicated by path.
 */
FALCON_FUNC CellRendererText::signal_edited( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "edited", (void*) &CellRendererText::on_edited, vm );
}


void CellRendererText::on_edited( GtkCellRendererText* obj, gchar* path,
                                  gchar* new_text, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "edited", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_edited", it ) )
            {
                printf(
                "[GtkCellRendererText::on_edited] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( UTF8String( path ) );
        vm->pushParam( UTF8String( new_text ) );
        vm->callItem( it, 2 );
    }
    while ( iter.hasCurrent() );
}


/*#
    @method set_fixed_height_from_font GtkCellRendererText
    @brief Sets the height of a renderer to explicitly be determined by the "font" and "y_pad" property set on it.
    @param number_of_rows Number of rows of text each cell renderer is allocated, or -1

    Further changes in these properties do not affect the height, so they must
    be accompanied by a subsequent call to this function. Using this function is
    unflexible, and should really only be used if calculating the size of a cell
    is too slow (ie, a massive number of cells displayed). If number_of_rows is
    -1, then the fixed height is unset, and the height is determined by the
    properties again.
 */
FALCON_FUNC CellRendererText::set_fixed_height_from_font( VMARG )
{
    Item* i_num = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_num || !i_num->isInteger() )
        throw_inv_params( "I" );
#endif
    gtk_cell_renderer_text_set_fixed_height_from_font( GET_CELLRENDERERTEXT( vm->self() ),
                                                       i_num->asInteger() );
}


} // Gtk
} // Falcon
