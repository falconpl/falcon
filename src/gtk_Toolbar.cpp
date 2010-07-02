/**
 *  \file gtk_Toolbar.cpp
 */

#include "gtk_Toolbar.hpp"

#include "gtk_Buildable.hpp"
#include "gtk_Orientable.hpp"
#include "gtk_ToolShell.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Toolbar::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Toolbar = mod->addClass( "GtkToolbar", &Toolbar::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkContainer" ) );
    c_Toolbar->getClassDef()->addInheritance( in );

    c_Toolbar->getClassDef()->factory( &Toolbar::factory );

    Gtk::MethodTab methods[] =
    {
    //{ "signal_focus_home_or_end",     &Toolbar::signal_focus_home_or_end },
    { "signal_orientation_changed",     &Toolbar::signal_orientation_changed },
    { "signal_popup_context_menu",      &Toolbar::signal_popup_context_menu },
    { "signal_style_changed",           &Toolbar::signal_style_changed },
    { "insert",                         &Toolbar::insert },
    { "get_item_index",                 &Toolbar::get_item_index },
    { "get_n_items",                    &Toolbar::get_n_items },
    //{ "get_nth_item",                   &Toolbar::get_nth_item },
    { "get_drop_index",                 &Toolbar::get_drop_index },
    { "set_drop_highlight_item",        &Toolbar::set_drop_highlight_item },
    { "set_show_arrow",                 &Toolbar::set_show_arrow },
    //{ "set_orientation",              &Toolbar::set_orientation },
    //{ "set_tooltips",                   &Toolbar::set_tooltips },
    { "unset_icon_size",                &Toolbar::unset_icon_size },
    { "get_show_arrow",                 &Toolbar::get_show_arrow },
    //{ "get_orientation",              &Toolbar::get_orientation },
    { "get_style",                      &Toolbar::get_style },
    { "get_icon_size",                  &Toolbar::get_icon_size },
    //{ "get_tooltips",                 &Toolbar::get_tooltips },
    { "get_relief_style",               &Toolbar::get_relief_style },
#if 0
    { "append_item",                    &Toolbar::append_item },
    { "prepend_item",                   &Toolbar::prepend_item },
    { "insert_item",                    &Toolbar::insert_item },
    { "append_space",                   &Toolbar::append_space },
    { "prepend_space",                  &Toolbar::prepend_space },
    { "insert_space",                   &Toolbar::insert_space },
    { "append_element",                 &Toolbar::append_element },
    { "prepend_element",                &Toolbar::prepend_element },
    { "insert_element",                 &Toolbar::insert_element },
    { "append_widget",                  &Toolbar::append_widget },
    { "prepend_widget",                 &Toolbar::prepend_widget },
    { "insert_widget",                  &Toolbar::insert_widget },
#endif
    { "set_style",                      &Toolbar::set_style },
    //{ "insert_stock",                   &Toolbar::insert_stock },
    { "set_icon_size",                  &Toolbar::set_icon_size },
    //{ "remove_space",                 &Toolbar::remove_space },
    { "unset_style",                    &Toolbar::unset_style },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Toolbar, meth->name, meth->cb );

    Gtk::Buildable::clsInit( mod, c_Toolbar );
#if GTK_CHECK_VERSION( 2, 16, 0 )
    Gtk::Orientable::clsInit( mod, c_Toolbar );
#endif
#if GTK_CHECK_VERSION( 2, 14, 0 )
    Gtk::ToolShell::clsInit( mod, c_Toolbar );
#endif
}


Toolbar::Toolbar( const Falcon::CoreClass* gen, const GtkToolbar* tbar )
    :
    Gtk::CoreGObject( gen, (GObject*) tbar )
{}


Falcon::CoreObject* Toolbar::factory( const Falcon::CoreClass* gen, void* tbar, bool )
{
    return new Toolbar( gen, (GtkToolbar*) tbar );
}


/*#
    @class GtkToolbar
    @brief Create bars of buttons and other widgets

    A toolbar can contain instances of a subclass of GtkToolItem. To add a
    GtkToolItem to the a toolbar, use gtk_toolbar_insert(). To remove an
    item from the toolbar use gtk_container_remove(). To add a button to
    the toolbar, add an instance of GtkToolButton.

    Toolbar items can be visually grouped by adding instances of
    GtkSeparatorToolItem to the toolbar. If a GtkSeparatorToolItem has
    the "expand" property set to TRUE and the "draw" property set to
    FALSE the effect is to force all following items to the end of the toolbar.

    Creating a context menu for the toolbar can be done by connecting to
    the "popup-context-menu" signal.

 */
FALCON_FUNC Toolbar::init( VMARG )
{
    NO_ARGS
    MYSELF;
    self->setObject( (GObject*) gtk_toolbar_new() );
}


//FALCON_FUNC Toolbar::signal_focus_home_or_end( VMARG );


/*#
    @method signal_orientation_changed GtkToolbar
    @brief Emitted when the orientation of the toolbar changes.
 */
FALCON_FUNC Toolbar::signal_orientation_changed( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "orientation_changed",
                             (void*) &Toolbar::on_orientation_changed, vm );
}


void Toolbar::on_orientation_changed( GtkToolbar* obj, GtkOrientation orient, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "orientation_changed", false );

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
                || !it.asObject()->getMethod( "on_orientation_changed", it ) )
            {
                printf(
                "[GtkToolbar::on_orientation_changed] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( (int64) orient );
        vm->callItem( it, 1 );
        iter.next();
    }
    while ( iter.hasCurrent() );
}


/*#
    @method signal_popup_context_menu GtkToolbar
    @brief Emitted when the user right-clicks the toolbar or uses the keybinding to display a popup menu.

    Application developers should handle this signal if they want to display
    a context menu on the toolbar. The context-menu should appear at the coordinates
    given by x and y. The mouse button number is given by the button parameter.
    If the menu was popped up using the keybaord, button is -1.
 */
FALCON_FUNC Toolbar::signal_popup_context_menu( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "popup_context_menu",
                             (void*) &Toolbar::on_popup_context_menu, vm );
}


void Toolbar::on_popup_context_menu( GtkToolbar* obj, gint x, gint y,
                                     gint button, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "popup_context_menu", false );

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
                || !it.asObject()->getMethod( "on_popup_context_menu", it ) )
            {
                printf(
                "[GtkToolbar::on_popup_context_menu] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( x );
        vm->pushParam( y );
        vm->pushParam( button );
        vm->callItem( it, 3 );
        iter.next();
    }
    while ( iter.hasCurrent() );
}


/*#
    @method signal_style_changed GtkToolbar
    @brief Emitted when the style of the toolbar changes.
 */
FALCON_FUNC Toolbar::signal_style_changed( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "style_changed",
                             (void*) &Toolbar::on_style_changed, vm );
}


void Toolbar::on_style_changed( GtkToolbar* obj, GtkToolbarStyle style, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "style_changed", false );

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
                || !it.asObject()->getMethod( "on_style_changed", it ) )
            {
                printf(
                "[GtkToolbar::on_style_changed] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( (int64) style );
        vm->callItem( it, 1 );
        iter.next();
    }
    while ( iter.hasCurrent() );
}


/*#
    @method insert GtkToolbar
    @brief Insert a GtkToolItem into the toolbar at position pos.
    @param item a GtkToolItem
    @param pos the position of the new item

    If pos is 0 the item is prepended to the start of the toolbar. If pos is
    negative, the item is appended to the end of the toolbar.
 */
FALCON_FUNC Toolbar::insert( VMARG )
{
    Item* i_titem = vm->param( 0 );
    Item* i_pos = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_titem || i_titem->isNil() || !i_titem->isObject()
        || !IS_DERIVED( i_titem, GtkToolItem )
        || !i_pos || i_pos->isNil() || !i_pos->isInteger() )
        throw_inv_params( "GtkToolItem,I" );
#endif
    GtkToolItem* titem = (GtkToolItem*) COREGOBJECT( i_titem )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_toolbar_insert( (GtkToolbar*)_obj, titem, i_pos->asInteger() );
}


/*#
    @method get_item_index GtkToolbar
    @brief Returns the position of item on the toolbar, starting from 0.
    @param item a GtkToolItem that is a child of toolbar
    @return The position of item on the toolbar

    It is an error if item is not a child of the toolbar.
 */
FALCON_FUNC Toolbar::get_item_index( VMARG )
{
    Item* i_titem = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_titem || i_titem->isNil() || !i_titem->isObject()
        || !IS_DERIVED( i_titem, GtkToolItem ) )
        throw_inv_params( "GtkToolItem" );
#endif
    GtkToolItem* titem = (GtkToolItem*) COREGOBJECT( i_titem )->getObject();
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_toolbar_get_item_index( (GtkToolbar*)_obj, titem ) );
}


/*#
    @method get_n_items GtkToolbar
    @brief Returns the number of items on the toolbar.
    @return the number of items on the toolbar
 */
FALCON_FUNC Toolbar::get_n_items( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_toolbar_get_n_items( (GtkToolbar*)_obj ) );
}


/*#
    @method get_nth_item GtkToolbar
    @brief Returns the n'th item on toolbar, or NULL if the toolbar does not contain an n'th item.
 */
//FALCON_FUNC Toolbar::get_nth_item( VMARG );


/*#
    @method get_drop_index GtkToolbar
    @brief Returns the position corresponding to the indicated point on toolbar.
    @param x x coordinate of a point on the toolbar
    @param y y coordinate of a point on the toolbar
    @return The position corresponding to the point (x, y) on the toolbar.

    This is useful when dragging items to the toolbar: this function returns the
    position a new item should be inserted.

    x and y are in toolbar coordinates.
 */
FALCON_FUNC Toolbar::get_drop_index( VMARG )
{
    Item* i_x = vm->param( 0 );
    Item* i_y = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_x || i_x->isNil() || !i_x->isInteger()
        || !i_y || i_y->isNil() || !i_y->isInteger() )
        throw_inv_params( "I,I" );
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_toolbar_get_drop_index( (GtkToolbar*)_obj,
                                            i_x->asInteger(), i_y->asInteger() ) );
}


/*#
    @method set_drop_highlight_item GtkToolbar
    @brief Highlights toolbar to give an idea of what it would look like if item was added to toolbar at the position indicated by index_.
    @param tool_item a GtkToolItem, or nil to turn of highlighting.
    @param index a position on toolbar

    If item is nil, highlighting is turned off. In that case index_ is ignored.

    The tool_item passed to this function must not be part of any widget hierarchy.
    When an item is set as drop highlight item it can not added to any widget
    hierarchy or used as highlight item for another toolbar.
 */
FALCON_FUNC Toolbar::set_drop_highlight_item( VMARG )
{
    Item* i_titem = vm->param( 0 );
    Item* i_idx = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_titem || !( i_titem->isNil() || ( i_titem->isObject()
        && IS_DERIVED( i_titem, GtkToolItem ) ) )
        || !i_idx || i_idx->isNil() || !i_idx->isInteger() )
        throw_inv_params( "GtkToolItem,I" );
#endif
    GtkToolItem* titem = i_titem->isNil() ? NULL :
                            (GtkToolItem*) COREGOBJECT( i_titem )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_toolbar_set_drop_highlight_item( (GtkToolbar*)_obj, titem, i_idx->asInteger() );
}


/*#
    @method set_show_arrow GtkToolbar
    @brief Sets whether to show an overflow menu when toolbar doesn't have room for all items on it.
    @param show_arrow Whether to show an overflow menu

    If TRUE, items that there are not room are available through an overflow menu.
 */
FALCON_FUNC Toolbar::set_show_arrow( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_toolbar_set_show_arrow( (GtkToolbar*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


#if 0 // deprecated
FALCON_FUNC Toolbar::set_orientation( VMARG );
FALCON_FUNC Toolbar::set_tooltips( VMARG );
#endif


/*#
    @method unset_icon_size GtkToolbar
    @brief Unsets toolbar icon size set with gtk_toolbar_set_icon_size(), so that user preferences will be used to determine the icon size.
 */
FALCON_FUNC Toolbar::unset_icon_size( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_toolbar_unset_icon_size( (GtkToolbar*)_obj );
}


/*#
    @method get_show_arrow GtkToolbar
    @brief Returns whether the toolbar has an overflow menu.
    @return TRUE if the toolbar has an overflow menu.
 */
FALCON_FUNC Toolbar::get_show_arrow( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_toolbar_get_show_arrow( (GtkToolbar*)_obj ) );
}


//FALCON_FUNC Toolbar::get_orientation( VMARG );


/*#
    @method get_style GtkToolbar
    @brief Retrieves whether the toolbar has text, icons, or both.
    @return the current style of toolbar (GtkToolbarStyle).
 */
FALCON_FUNC Toolbar::get_style( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_toolbar_get_style( (GtkToolbar*)_obj ) );
}


/*#
    @method get_icon_size GtkToolbar
    @brief Retrieves the icon size for the toolbar.
    @return the current icon size for the icons on the toolbar (GtkIconSize).
 */
FALCON_FUNC Toolbar::get_icon_size( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_toolbar_get_icon_size( (GtkToolbar*)_obj ) );
}


//FALCON_FUNC Toolbar::get_tooltips( VMARG );


/*#
    @method get_relief_style GtkToolbar
    @brief Returns the relief style of buttons on toolbar.
    @return The relief style of buttons on toolbar (GtkReliefStyle).
 */
FALCON_FUNC Toolbar::get_relief_style( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_toolbar_get_icon_size( (GtkToolbar*)_obj ) );
}


#if 0 // deprecated
FALCON_FUNC Toolbar::append_item( VMARG );
FALCON_FUNC Toolbar::prepend_item( VMARG );
FALCON_FUNC Toolbar::insert_item( VMARG );
FALCON_FUNC Toolbar::append_space( VMARG );
FALCON_FUNC Toolbar::prepend_space( VMARG );
FALCON_FUNC Toolbar::insert_space( VMARG );
FALCON_FUNC Toolbar::append_element( VMARG );
FALCON_FUNC Toolbar::prepend_element( VMARG );
FALCON_FUNC Toolbar::insert_element( VMARG );
FALCON_FUNC Toolbar::append_widget( VMARG );
FALCON_FUNC Toolbar::prepend_widget( VMARG );
FALCON_FUNC Toolbar::insert_widget( VMARG );
#endif


/*#
    @method set_style GtkToolbar
    @brief Alters the view of toolbar to display either icons only, text only, or both.
    @param style the new style for toolbar (GtkToolbarStyle).
 */
FALCON_FUNC Toolbar::set_style( VMARG )
{
    Item* i_style = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_style || i_style->isNil() || !i_style->isInteger() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_toolbar_set_style( (GtkToolbar*)_obj, (GtkToolbarStyle) i_style->asInteger() );
}


//FALCON_FUNC Toolbar::insert_stock( VMARG );


/*#
    @method set_icon_size GtkToolbar
    @brief This function sets the size of stock icons in the toolbar.
    @param icon_size The GtkIconSize that stock icons in the toolbar shall have

    You can call it both before you add the icons and after they've been added.
    The size you set will override user preferences for the default icon size.

    This should only be used for special-purpose toolbars, normal application
    toolbars should respect the user preferences for the size of icons.
 */
FALCON_FUNC Toolbar::set_icon_size( VMARG )
{
    Item* i_size = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_size || i_size->isNil() || !i_size->isInteger() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_toolbar_set_icon_size( (GtkToolbar*)_obj, (GtkIconSize) i_size->asInteger() );
}


//FALCON_FUNC Toolbar::remove_space( VMARG );


/*#
    @method unset_style GtkToolbar
    @brief Unsets a toolbar style set with gtk_toolbar_set_style(), so that user preferences will be used to determine the toolbar style.
 */
FALCON_FUNC Toolbar::unset_style( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_toolbar_unset_style( (GtkToolbar*)_obj );
}


} // Gtk
} // Falcon
