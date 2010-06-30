/**
 *  \file gtk_CellRendererAccel.cpp
 */

#include "gtk_CellRendererAccel.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void CellRendererAccel::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_CellRendererAccel = mod->addClass( "GtkCellRendererAccel", &CellRendererAccel::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkCellRendererText" ) );
    c_CellRendererAccel->getClassDef()->addInheritance( in );

    //c_CellRendererAccel->setWKS( true );
    c_CellRendererAccel->getClassDef()->factory( &CellRendererAccel::factory );

    mod->addClassMethod( c_CellRendererAccel,
                         "signal_accel_cleared",
                         &CellRendererAccel::signal_accel_cleared );
    mod->addClassMethod( c_CellRendererAccel,
                         "signal_accel_edited",
                         &CellRendererAccel::signal_accel_edited );
}


CellRendererAccel::CellRendererAccel( const Falcon::CoreClass* gen, const GtkCellRendererAccel* renderer )
    :
    Gtk::CoreGObject( gen, (GObject*) renderer )
{}


Falcon::CoreObject* CellRendererAccel::factory( const Falcon::CoreClass* gen, void* renderer, bool )
{
    return new CellRendererAccel( gen, (GtkCellRendererAccel*) renderer );
}


/*#
    @class GtkCellRendererAccel
    @brief Renders a keyboard accelerator in a cell

    GtkCellRendererAccel displays a keyboard accelerator (i.e. a key combination
    like <Control>-a). If the cell renderer is editable, the accelerator can be
    changed by simply typing the new combination.
 */
FALCON_FUNC CellRendererAccel::init( VMARG )
{
    NO_ARGS
    MYSELF;
    self->setGObject( (GObject*) gtk_cell_renderer_accel_new() );
}


/*#
    @method signal_accel_cleared GtkCellRendererAccel
    @brief Gets emitted when the user has removed the accelerator.
 */
FALCON_FUNC CellRendererAccel::signal_accel_cleared( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "accel_cleared", (void*) &CellRendererAccel::on_accel_cleared, vm );
}


void CellRendererAccel::on_accel_cleared( GtkCellRendererAccel* obj, gchar* path, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "accel_cleared", false );

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
                || !it.asObject()->getMethod( "on_accel_cleared", it ) )
            {
                printf(
                "[GtkCellRendererAccel::on_accel_cleared] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( UTF8String( path ) );
        vm->callItem( it, 1 );
    }
    while ( iter.hasCurrent() );
}


/*#
    @method signal_accel_edited GtkCellRendererAccel
    @brief Gets emitted when the user has selected a new accelerator.
 */
FALCON_FUNC CellRendererAccel::signal_accel_edited( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "accel_edited", (void*) &CellRendererAccel::on_accel_edited, vm );
}


void CellRendererAccel::on_accel_edited( GtkCellRendererAccel* obj, gchar* path, guint key,
                                GdkModifierType mode, guint keycode, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "accel_edited", false );

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
                || !it.asObject()->getMethod( "on_accel_edited", it ) )
            {
                printf(
                "[GtkCellRendererAccel::on_accel_edited] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( UTF8String( path ) );
        vm->pushParam( key );
        vm->pushParam( (int64) mode );
        vm->pushParam( keycode );
        vm->callItem( it, 4 );
    }
    while ( iter.hasCurrent() );
}


} // Gtk
} // Falcon
